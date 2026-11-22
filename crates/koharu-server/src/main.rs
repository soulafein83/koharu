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
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/ocr", post(ocr))
        .route("/detect", post(detect))
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
