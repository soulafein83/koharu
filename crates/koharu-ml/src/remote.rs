//! Serializable form of `koharu_layout_highres::Layout`, for inference over
//! HTTP: a server runs detection and sends this back; the client turns it
//! back into a `Layout`.

use crate::koharu_layout_highres::{Bubble, Kind, Layout, Lettering, Mask, Panel};
use kurbo::BezPath;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteMask {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl From<&Mask> for RemoteMask {
    fn from(m: &Mask) -> Self {
        Self {
            x: m.x,
            y: m.y,
            width: m.width,
            height: m.height,
            pixels: m.pixels.clone(),
        }
    }
}

impl From<RemoteMask> for Mask {
    fn from(m: RemoteMask) -> Self {
        Self {
            x: m.x,
            y: m.y,
            width: m.width,
            height: m.height,
            pixels: m.pixels,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteLettering {
    pub kind: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub mask: RemoteMask,
    pub bubble: Option<usize>,
    pub panel: Option<usize>,
    pub room: Option<String>,
}

impl From<&Lettering> for RemoteLettering {
    fn from(l: &Lettering) -> Self {
        Self {
            kind: match l.kind {
                Kind::Text => "text".to_owned(),
                Kind::Onomatopoeia => "onomatopoeia".to_owned(),
            },
            score: l.score,
            bbox: l.bbox,
            mask: RemoteMask::from(&l.mask),
            bubble: l.bubble,
            panel: l.panel,
            room: l.room.as_ref().map(|r| r.to_svg()),
        }
    }
}

impl From<RemoteLettering> for Lettering {
    fn from(l: RemoteLettering) -> Self {
        let kind = if l.kind == "onomatopoeia" {
            Kind::Onomatopoeia
        } else {
            Kind::Text
        };
        let room = l.room.as_deref().and_then(|s| BezPath::from_svg(s).ok());
        Self {
            kind,
            score: l.score,
            bbox: l.bbox,
            mask: Mask::from(l.mask),
            bubble: l.bubble,
            panel: l.panel,
            room,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteBubble {
    pub score: f32,
    pub bbox: [f32; 4],
    pub mask: RemoteMask,
    pub outline: String,
    pub lettering: Vec<usize>,
    pub panel: Option<usize>,
}

impl From<&Bubble> for RemoteBubble {
    fn from(b: &Bubble) -> Self {
        Self {
            score: b.score,
            bbox: b.bbox,
            mask: RemoteMask::from(&b.mask),
            outline: b.outline.to_svg(),
            lettering: b.lettering.clone(),
            panel: b.panel,
        }
    }
}

impl From<RemoteBubble> for Bubble {
    fn from(b: RemoteBubble) -> Self {
        let outline = BezPath::from_svg(&b.outline).unwrap_or_default();
        Self {
            score: b.score,
            bbox: b.bbox,
            mask: Mask::from(b.mask),
            outline,
            lettering: b.lettering,
            panel: b.panel,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemotePanel {
    pub score: f32,
    pub bbox: [f32; 4],
    pub mask: RemoteMask,
}

impl From<&Panel> for RemotePanel {
    fn from(p: &Panel) -> Self {
        Self {
            score: p.score,
            bbox: p.bbox,
            mask: RemoteMask::from(&p.mask),
        }
    }
}

impl From<RemotePanel> for Panel {
    fn from(p: RemotePanel) -> Self {
        Self {
            score: p.score,
            bbox: p.bbox,
            mask: Mask::from(p.mask),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteLayout {
    pub width: u32,
    pub height: u32,
    pub lettering: Vec<RemoteLettering>,
    pub bubbles: Vec<RemoteBubble>,
    pub panels: Vec<RemotePanel>,
}

impl From<&Layout> for RemoteLayout {
    fn from(l: &Layout) -> Self {
        Self {
            width: l.width,
            height: l.height,
            lettering: l.lettering.iter().map(RemoteLettering::from).collect(),
            bubbles: l.bubbles.iter().map(RemoteBubble::from).collect(),
            panels: l.panels.iter().map(RemotePanel::from).collect(),
        }
    }
}

impl From<RemoteLayout> for Layout {
    fn from(l: RemoteLayout) -> Self {
        Self {
            width: l.width,
            height: l.height,
            lettering: l.lettering.into_iter().map(Lettering::from).collect(),
            bubbles: l.bubbles.into_iter().map(Bubble::from).collect(),
            panels: l.panels.into_iter().map(Panel::from).collect(),
        }
    }
}
