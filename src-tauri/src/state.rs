//! Shared application state held by Tauri's `.manage()`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::inference::{InferenceBackend, ModelInfo};

#[derive(Default)]
pub struct AppState {
    /// The active engine, if a model has been loaded.
    pub engine: RwLock<Option<Arc<dyn InferenceBackend>>>,
    /// Metadata about the loaded model for display.
    pub model: RwLock<Option<ModelInfo>>,
    /// Set to `true` to ask the in-flight generation to stop early.
    pub cancel: Arc<AtomicBool>,
}

impl AppState {
    /// Clone out the active backend without holding the lock during generation.
    pub async fn backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        self.engine.read().await.clone()
    }
}
