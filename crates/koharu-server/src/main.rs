use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Multipart, State},
    routing::{get, post},
};
use image::DynamicImage;
use koharu_ml::hayai_ocr::HayaiOcr;
use koharu_ml::koharu_layout_highres::{KoharuLayoutHighres, Options};
use koharu_runtime;
use serde::Serialize;
use serde_json::json;
use tokio::sync::Mutex;
use tracing::info;

#[derive(Clone)]
struct AppState {
    ocr: Arc<Mutex<Option<HayaiOcr>>>,
    detector: Arc<Mutex<Option<koharu_ml::koharu_layout_highres::KoharuLayoutHighres>>>,
    inpainter: Arc<Mutex<Option<(String, InpainterInstance)>>>,
}

#[derive(Serialize)]
struct OcrResponse {
    texts: Vec<String>,
    language: &'static str,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let state = AppState {
        ocr: Arc::new(Mutex::new(None)),
        detector: Arc::new(Mutex::new(None)),
        inpainter: Arc::new(Mutex::new(None)),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/ocr", post(ocr))
        .route("/detect", post(detect))
        .route("/inpaint", post(inpaint_handler))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 13306));
    info!("koharu-server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok", "version": env!("CARGO_PKG_VERSION")}))
}

async fn ocr(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<OcrResponse>, (axum::http::StatusCode, String)> {
    // Collect uploaded images.
    let mut images: Vec<DynamicImage> = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?
    {
        let name = field.name().unwrap_or("").to_owned();
        if name != "image" && name != "images" {
            continue;
        }
        let bytes = field
            .bytes()
            .await
            .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?;
        let img = image::load_from_memory(&bytes)
            .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?;
        images.push(img);
    }
    if images.is_empty() {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "no images uploaded (field 'image')".to_owned(),
        ));
    }

    // Lazy-load model.
    let mut guard = state.ocr.lock().await;
    if guard.is_none() {
        let device = koharu_runtime::device()
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let model = HayaiOcr::load(device)
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        *guard = Some(model);
    }
    let model = guard.as_ref().unwrap();
    let texts = model
        .inference(&images)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(OcrResponse {
        texts,
        language: "und",
    }))
}


#[derive(serde::Deserialize)]
struct DetectQuery {
    text: Option<f32>,
    onomatopoeia: Option<f32>,
    bubble: Option<f32>,
    panel: Option<f32>,
}

async fn detect(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<DetectQuery>,
    mut multipart: Multipart,
) -> Result<Json<koharu_ml::remote::RemoteLayout>, (axum::http::StatusCode, String)> {
    let mut image: Option<image::DynamicImage> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?
    {
        let bytes = field
            .bytes()
            .await
            .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?;
        image = Some(
            image::load_from_memory(&bytes)
                .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?,
        );
        break;
    }
    let image = image.ok_or((
        axum::http::StatusCode::BAD_REQUEST,
        "no image uploaded".to_owned(),
    ))?;

    let mut guard = state.detector.lock().await;
    if guard.is_none() {
        let device = koharu_runtime::device()
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let model = KoharuLayoutHighres::load(device)
            .await
            .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        *guard = Some(model);
    }
    let model = guard.as_ref().unwrap();
    let options = Options {
        text: q.text,
        onomatopoeia: q.onomatopoeia,
        bubble: q.bubble,
        panel: q.panel,
    };
    let detected = model
        .inference(&image, &options)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let pixels = image.to_rgb8();
    let layout = detected.layout(&pixels);
    Ok(Json(koharu_ml::remote::RemoteLayout::from(&layout)))
}

#[derive(Debug)]
enum InpainterInstance {
    Lama(koharu_ml::lama::Lama),
    Flux(koharu_ml::flux3_klein::Flux3Klein),
    Qwen(koharu_ml::qwen_image::QwenImage),
}

impl InpainterInstance {
    async fn load(model: &str, device: &'static koharu_runtime::Device) -> Result<Self, String> {
        match model {
            "lama" => Ok(Self::Lama(
                koharu_ml::lama::Lama::load(device)
                    .await
                    .map_err(|e| e.to_string())?,
            )),
            "flux3-klein" => Ok(Self::Flux(
                koharu_ml::flux3_klein::Flux3Klein::load(device)
                    .await
                    .map_err(|e| e.to_string())?,
            )),
            "qwen-image" => Ok(Self::Qwen(
                koharu_ml::qwen_image::QwenImage::load(device)
                    .await
                    .map_err(|e| e.to_string())?,
            )),
            _ => Err(format!("unknown model: {}", model)),
        }
    }

    fn infer(&self, image: &image::RgbImage, mask: &image::GrayImage, prompt: &str) -> Result<image::RgbImage, String> {
        match self {
            Self::Lama(m) => {
                let out = m
                    .inference(&image::DynamicImage::ImageRgb8(image.clone()), mask)
                    .map_err(|e| e.to_string())?;
                Ok(out)
            }
            Self::Flux(m) => {
                let edited = m
                    .generate(
                        &[image::DynamicImage::ImageRgb8(image.clone())],
                        prompt,
                        &koharu_ml::flux3_klein::GenerateOptions::default(),
                    )
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .next()
                    .ok_or_else(|| "no image".to_owned())?;
                Ok(edited)
            }
            Self::Qwen(m) => {
                let side = (image.width() as f64 * image.height() as f64).sqrt();
                let opts = koharu_ml::qwen_image::Options {
                    output_resolution: side.min(1024.0).round() as u32,
                    ..Default::default()
                };
                let edited = m
                    .generate(prompt, &[image::DynamicImage::ImageRgb8(image.clone())], &opts)
                    .map_err(|e| e.to_string())?
                    .into_iter()
                    .next()
                    .ok_or_else(|| "no image".to_owned())?;
                Ok(image::DynamicImage::ImageRgba8(edited).to_rgb8())
            }
        }
    }

    fn generative(&self) -> bool {
        matches!(self, Self::Flux(_) | Self::Qwen(_))
    }
}

async fn inpaint_handler(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<axum::response::Response, (axum::http::StatusCode, String)> {
    use axum::body::Body;
    use axum::http::header;
    use axum::response::Response;

    let mut image: Option<image::DynamicImage> = None;
    let mut mask: Option<image::DynamicImage> = None;
    let mut model = String::from("lama");
    let mut prompt = String::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?
    {
        match field.name().unwrap_or("") {
            "image" => {
                let bytes = field.bytes().await.map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?;
                image = Some(image::load_from_memory(&bytes).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?);
            }
            "mask" => {
                let bytes = field.bytes().await.map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?;
                mask = Some(image::load_from_memory(&bytes).map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?);
            }
            "model" => {
                model = field.text().await.unwrap_or_default();
            }
            "prompt" => {
                prompt = field.text().await.unwrap_or_default();
            }
            _ => {}
        }
    }
    let image = image.ok_or((axum::http::StatusCode::BAD_REQUEST, "no image".to_owned()))?;
    let mask = mask.ok_or((axum::http::StatusCode::BAD_REQUEST, "no mask".to_owned()))?;
    let image_rgb = image.to_rgb8();
    let mask_gray = mask.to_luma8();

    // cache
    let mut guard = state.inpainter.lock().await;
    if guard.as_ref().map(|(m, _)| m != &model).unwrap_or(true) {
        let device = koharu_runtime::device().await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        let inst = InpainterInstance::load(&model, device).await.map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
        *guard = Some((model.clone(), inst));
    }
    let inst = &guard.as_ref().unwrap().1;
    // Run inference directly (may block runtime briefly)
    let out = inst.infer(&image_rgb, &mask_gray, &prompt).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let mut buf = Vec::new();
    image::DynamicImage::ImageRgb8(out)
        .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let resp = Response::builder()
        .header(header::CONTENT_TYPE, "image/png")
        .body(Body::from(buf))
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(resp)
}
