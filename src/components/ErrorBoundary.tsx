import React from "react";

type Props = { children: React.ReactNode };
type State = { error: Error | null };

/**
 * Last-resort guard: catches any render-time exception in the app tree and
 * shows a recovery screen instead of a blank white window. Deliberately
 * bilingual and context-free — the i18n provider itself could be implicated in
 * the crash, so the fallback must not depend on any app context.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Keep the stack in the console/devtools for diagnosis; nothing is sent off
    // the machine.
    console.error("Unhandled render error:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="crash-screen">
        <div className="crash-card">
          <div className="crash-title">应用出现异常 · Something went wrong</div>
          <p className="crash-msg">
            Chaty 遇到了一个意外错误，但你的对话已经保存在本地、不会丢失。
            <br />
            Chaty hit an unexpected error. Your conversations are safe on disk.
          </p>
          <pre className="crash-detail">{error.message || String(error)}</pre>
          <div className="crash-actions">
            <button className="crash-btn" onClick={() => window.location.reload()}>
              重新加载 · Reload
            </button>
          </div>
        </div>
      </div>
    );
  }
}
