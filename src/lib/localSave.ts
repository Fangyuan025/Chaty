/** Writing to localStorage can throw — a full quota, a browser with site data
 *  turned off, private mode. Every one of those calls sits in an effect, and an
 *  effect that throws takes the interface down to the error boundary: the app
 *  would be replaced by its crash page because a window width could not be
 *  remembered. Remembering is best-effort by nature, so it fails quietly and
 *  says so in the console. */
export function localSave(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch (e) {
    console.warn(`could not remember ${key}:`, e);
    return false;
  }
}
