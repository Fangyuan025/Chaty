/** Whether an Enter is the input method's: the key that commits a
 *  composition (pinyin typed out as letters, a candidate picked) arrives as
 *  an Enter keydown too, and taken as the user's it sent a half-typed
 *  message, renamed a chat to its pinyin or ran a command mid-word. WebKit
 *  marks it isComposing, or with keyCode 229 when it fires after the
 *  composition has ended. */
export function imeEnter(e: { nativeEvent: KeyboardEvent; keyCode: number }): boolean {
  return e.nativeEvent.isComposing || e.keyCode === 229;
}
