//! Detection: the page's panels, balloons, and blocks in reading order, in
//! place of the ones there, whoever made them. A block gets its source's
//! frame, tilt, and direction, and its mask.

mod layers;
mod model;
mod tree;

use koharu_ml::koharu_layout_highres::{Options, Typography};
use koharu_scene::{Content, Layer, Op};
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

pub use self::model::Detector;
use super::{Task, key};
use crate::{PipelineConfig, Stage};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[derive(specta::Type)]
pub struct DetectorConfig {
    pub model: Detector,
    /// Each class's lowest confidence a detection keeps, from 0 to 1, or the
    /// model's own.
    #[serde(deserialize_with = "threshold")]
    #[specta(type = Option<f32>)]
    pub text: Option<f32>,
    #[serde(deserialize_with = "threshold")]
    #[specta(type = Option<f32>)]
    pub onomatopoeia: Option<f32>,
    #[serde(deserialize_with = "threshold")]
    #[specta(type = Option<f32>)]
    pub bubble: Option<f32>,
    #[serde(deserialize_with = "threshold")]
    #[specta(type = Option<f32>)]
    pub panel: Option<f32>,
    /// Base URL of a remote inference server. When set, detection runs there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[specta(type = Option<String>)]
    pub remote: Option<String>,
}

fn threshold<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f32>, D::Error> {
    let threshold = Option::<f32>::deserialize(deserializer)?;
    match threshold {
        Some(value) if !(0.0..=1.0).contains(&value) => Err(D::Error::custom(format!(
            "a detection threshold of {value} isn't between 0 and 1"
        ))),
        _ => Ok(threshold),
    }
}

pub(crate) async fn detect(task: &Task, config: &PipelineConfig) -> anyhow::Result<()> {
    let member = &task.members[0];
    let read = member.clone();
    let scan = tokio::task::spawn_blocking(move || {
        let _span =
            tracing::info_span!("read", stage = %Stage::Detect, page = read.index + 1).entered();
        read.scan()
    })
    .await??;
    let Some((_, scan)) = scan else {
        return Ok(());
    };
    let remote = config.detector.remote.clone();
    let det_config = config.detector.clone();
    let name = det_config.model.to_string();
    let page = member.page;

    // Obtain the layout either from a remote server or locally.
    let (scan, layout): (image::DynamicImage, koharu_ml::koharu_layout_highres::Layout) =
        if let Some(base) = remote {
            // send scan PNG to remote /detect
            let mut buf = Vec::new();
            let png_bytes = {
                let mut c = std::io::Cursor::new(&mut buf);
                scan.write_to(&mut c, image::ImageFormat::Png)
                    .map_err(|e| anyhow::anyhow!(e))?;
                buf
            };
            let client = reqwest::Client::new();
            let mut url = format!("{}/detect", base.trim_end_matches('/'));
            let mut qs = Vec::new();
            if let Some(v) = det_config.text { qs.push(format!("text={}", v)); }
            if let Some(v) = det_config.onomatopoeia { qs.push(format!("onomatopoeia={}", v)); }
            if let Some(v) = det_config.bubble { qs.push(format!("bubble={}", v)); }
            if let Some(v) = det_config.panel { qs.push(format!("panel={}", v)); }
            if !qs.is_empty() { url = format!("{}?{}", url, qs.join("&")); }
            let form = reqwest::multipart::Form::new().part(
                "image",
                reqwest::multipart::Part::bytes(png_bytes)
                    .file_name("page.png")
                    .mime_str("image/png")
                    .map_err(|e| anyhow::anyhow!(e))?,
            );
            let resp = client
                .post(&url)
                .multipart(form)
                .send()
                .await
                .map_err(|e| anyhow::anyhow!(e))?;
            let status = resp.status();
            let body: serde_json::Value = resp.json().await.map_err(|e| anyhow::anyhow!(e))?;
            if !status.is_success() {
                anyhow::bail!("remote detect failed: {} {}", status, body);
            }
            let remote_layout: koharu_ml::remote::RemoteLayout =
                serde_json::from_value(body).map_err(|e| anyhow::anyhow!(e))?;
            (scan, koharu_ml::koharu_layout_highres::Layout::from(remote_layout))
        } else {
            let key = key(Stage::Detect, config).expect("a detector is Koharu's");
            let cfg = det_config.clone();
            let options = Options {
                text: cfg.text,
                onomatopoeia: cfg.onomatopoeia,
                bubble: cfg.bubble,
                panel: cfg.panel,
            };
            let (scan, detected) = task
                .inference(&[member], move |models| {
                    let detector = models.get(&key, |device| model::load(cfg.model, device))?;
                    let detected = detector.inference(&scan, &options)?;
                    Ok((scan, detected))
                })
                .await?;
            let pixels = scan.to_rgb8();
            let layout = detected.layout(&pixels);
            (scan, layout)
        };

    let additions = tokio::task::spawn_blocking(move || {
        let pixels = scan.to_rgb8();
        let _span = tracing::info_span!("typography").entered();
        let typography: Vec<Option<Typography>> = layout
            .lettering
            .par_iter()
            .map(|lettering| lettering.typography(&pixels))
            .collect();
        layers::additions(&layout, &typography, page, &name)
    })
    .await??;
    member
        .apply(Stage::Detect, move |project| {
            let Ok(page) = project.page(page) else {
                return Ok(Vec::new());
            };
            let mut ops = Vec::new();
            removals(&page.layers, &mut ops);
            ops.extend(additions);
            Ok(ops)
        })
        .await
}

/// Removes the page's panels, balloons, and blocks, each with what's inside
/// it, but a locked one, which the user keeps.
fn removals(layers: &[Layer], ops: &mut Vec<Op>) {
    for layer in layers.iter().filter(|layer| !layer.locked) {
        match layer.content {
            Content::Panel(_) | Content::Balloon(_) | Content::Block(_) => {
                ops.push(Op::RemoveLayer {
                    layer: layer.id.into(),
                })
            }
            _ => removals(&layer.children, ops),
        }
    }
}
