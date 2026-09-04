//! Shared application state held by Tauri's `.manage()`.

use std::sync::atomic::AtomicBool;
use std::sync::Mutex;
use std::sync::Arc;

use tokio::sync::RwLock;

use tauri::ipc::Channel;
use crate::inference::{InferenceBackend, ModelInfo, StreamEvent};

#[derive(Default)]
pub struct AppState {
    /// The active engine, if a model has been loaded.
    pub engine: RwLock<Option<Arc<dyn InferenceBackend>>>,
    /// Metadata about the loaded model for display.
    pub model: RwLock<Option<ModelInfo>>,
    /// Set to `true` to ask the in-flight generation to stop early.
    pub cancel: Arc<AtomicBool>,
    /// The turn currently being generated, if any.
    ///
    /// A turn outlives the page that asked for it: the webview can be replaced
    /// mid-generation, taking the JS context — and with it the only copy of a
    /// long answer — away with it. Keeping the turn here means the work is not
    /// bound to that page. A page that arrives afterwards asks for this, is
    /// handed everything generated so far, and takes over receiving the rest.
    pub live: Mutex<Option<LiveTurn>>,
}

/// A generation in progress, held by the app rather than by a page.
pub struct LiveTurn {
    /// Where the reply belongs, so it can be written down and found again.
    pub conversation_id: String,
    pub message_id: String,
    /// Everything generated so far.
    pub text: Arc<Mutex<String>>,
    /// Whoever is currently receiving events — swapped when a page takes over,
    /// `None` while nobody is listening, which is not a reason to stop.
    pub listener: Arc<Mutex<Option<Channel<StreamEvent>>>>,
}

impl AppState {
    /// Clone out the active backend without holding the lock during generation.
    pub async fn backend(&self) -> Option<Arc<dyn InferenceBackend>> {
        self.engine.read().await.clone()
    }
}
