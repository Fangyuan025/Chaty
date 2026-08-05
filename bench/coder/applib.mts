/** Shared grading for app-delivery benches: stack detection, compile truth,
 * bundle + launch checks. Extracted from calrepro (rounds 1-24) so the
 * multi-stack matrix bench grades identically. */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";

export function walk(dir: string, depth: number, hit: (p: string) => void) {
  if (depth < 0) return;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    if (e.name.startsWith(".")) continue;
    const p = path.join(dir, e.name);
    hit(p);
    if (e.isDirectory()) walk(p, depth - 1, hit);
  }
}

/** Owner bar v2: compile green is only step one — the deliverable is a
 *  packaged .app that LAUNCHES and stays alive for 4 seconds. */
export async function grade(ws: string): Promise<{ compiles: boolean; packaged: boolean; runs: boolean; errors: string[]; how: string }> {
  const c = gradeCompile(ws);
  if (!c.compiles) return { ...c, packaged: false, runs: false };
  // Find delivered .app bundles — excluding node_modules (the stock
  // Electron.app inside the dependency is NOT the delivery) and .build.
  const bundles: string[] = [];
  walk(ws, 6, (p) => {
    if (
      p.endsWith(".app") &&
      !p.includes("node_modules") &&
      !p.includes("/.build/") &&
      statSync(p).isDirectory() &&
      existsSync(path.join(p, "Contents", "MacOS"))
    )
      bundles.push(p);
  });
  if (!bundles.length) {
    return { ...c, packaged: false, runs: false, errors: ["compile green but no packaged .app bundle delivered"], how: c.how + "+no-bundle" };
  }
  // Launch check: the bundle's main binary must survive 4 seconds.
  const macos = path.join(bundles[0], "Contents", "MacOS");
  const bin = readdirSync(macos).find((f) => { try { return (statSync(path.join(macos, f)).mode & 0o111) !== 0; } catch { return false; } });
  if (!bin) return { ...c, packaged: true, runs: false, errors: ["bundle has no executable in Contents/MacOS"], how: c.how + "+launch" };
  const child = spawn(path.join(macos, bin), [], { stdio: "ignore", detached: true });
  let dead = false;
  child.on("exit", () => { dead = true; });
  await new Promise((r) => setTimeout(r, 4000));
  const runs = !dead;
  try { if (child.pid) process.kill(-child.pid); } catch { try { child.kill(); } catch { /* already gone */ } }
  return {
    ...c,
    packaged: true,
    runs,
    errors: runs ? c.errors : [`launch check failed: ${path.basename(bundles[0])} exited within 4s`],
    how: c.how + "+launch",
  };
}

function gradeCompile(ws: string): { compiles: boolean; errors: string[]; how: string } {
  let xcodeproj: string | undefined;
  let packageDir: string | undefined;
  const swifts: string[] = [];
  walk(ws, 4, (p) => {
    if (p.endsWith(".xcodeproj") && statSync(p).isDirectory()) xcodeproj ??= p;
    // Track the manifest's own directory: `swift build` must run THERE.
    // Running it at the ws root walks UP the parent chain for a manifest —
    // round 17 "passed" by silently building a stale package that round 1
    // had left in the shared temp ROOT, two directories above the delivery.
    if (path.basename(p) === "Package.swift") packageDir ??= path.dirname(p);
    // Package manifests need the SwiftPM toolchain; bare swiftc reports a
    // phantom "no such module 'PackageDescription'" on fine projects.
    if (p.endsWith(".swift") && !/Tests\//.test(p) && !/\/Package(@swift-[^/]*)?\.swift$/.test(p))
      swifts.push(p);
  });
  const run = (bin: string, args: string[]): { code: number; out: string } => {
    try {
      const out = execFileSync(bin, args, { cwd: ws, timeout: 300_000, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
      return { code: 0, out };
    } catch (e) {
      const err = e as { status?: number; stdout?: string; stderr?: string };
      return { code: err.status ?? 1, out: `${err.stdout ?? ""}\n${err.stderr ?? ""}` };
    }
  };
  const errLines = (out: string) =>
    [...new Set(out.split("\n").filter((l) => / error: |^error: /.test(l)))].slice(0, 20);
  // Substance floor (round 13: a cleanup rm -rf left ONE typecheck-clean
  // file and the compile check waved the hollow tree through): an app
  // delivery needs an entry point and more than a fragment of source.
  const hollow = (why: string) => ({ compiles: false, errors: [why], how: "substance" });
  if (swifts.length) {
    const hasMain = swifts.some((p) => {
      try { return readFileSync(p, "utf8").includes("@main"); } catch { return false; }
    });
    if (!hasMain) return hollow("no @main entry point in delivered Swift sources");
    // Volume, not file COUNT: a complete calculator fits in two files (calc2
    // false-flag), while the round-13 rm-wiped tree was a single 80-line
    // fragment with no entry point — @main plus a modest line floor tells
    // them apart.
    const totalLines = swifts.reduce((n, p) => {
      try { return n + readFileSync(p, "utf8").split("\n").length; } catch { return n; }
    }, 0);
    if (totalLines < 100) return hollow(`only ${totalLines} Swift source lines delivered — not an app`);
  }
  if (xcodeproj) {
    const r = run("xcodebuild", [
      "-project", xcodeproj, "-alltargets", "-configuration", "Debug",
      "build", "CODE_SIGNING_ALLOWED=NO",
    ]);
    return { compiles: r.code === 0 && r.out.includes("BUILD SUCCEEDED"), errors: errLines(r.out), how: "xcodebuild" };
  }
  if (packageDir && existsSync(path.join(packageDir, "Sources"))) {
    const r = run("swift", ["build", "--package-path", packageDir]);
    return { compiles: r.code === 0, errors: errLines(r.out), how: "swift build" };
  }
  if (swifts.length) {
    const r = run("swiftc", ["-typecheck", ...swifts]);
    return { compiles: r.code === 0, errors: errLines(r.out), how: "swiftc -typecheck" };
  }
  // Node/TS delivery (the model may legitimately pick Electron/Vue/Tauri for
  // a "mac desktop app"): find the project dir (round 18 shipped in a
  // subdirectory and the root-only check called a green delivery "nothing"),
  // then prefer the project's OWN build script — that is the honest "does it
  // compile" for whatever stack it chose; tsc/vite are the fallback.
  let pkgDir: string | undefined;
  walk(ws, 3, (p) => {
    if (path.basename(p) === "package.json" && !p.includes("node_modules")) pkgDir ??= path.dirname(p);
  });
  if (pkgDir) {
    const dir = pkgDir;
    const runIn = (bin: string, args: string[]) => {
      try {
        const out = execFileSync(bin, args, { cwd: dir, timeout: 300_000, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
        return { code: 0, out };
      } catch (e) {
        const err = e as { status?: number; stdout?: string; stderr?: string };
        return { code: err.status ?? 1, out: `${err.stdout ?? ""}\n${err.stderr ?? ""}` };
      }
    };
    const tsErr = (out: string) => [...new Set(out.split("\n").filter((l) => /error TS\d+:/.test(l)))].slice(0, 20);
    if (!existsSync(path.join(dir, "node_modules"))) {
      const i = runIn("npm", ["install", "--no-audit", "--no-fund"]);
      if (i.code !== 0) return { compiles: false, errors: ["npm install failed", ...i.out.split("\n").slice(-5)], how: "npm install" };
    }
    const scripts = (JSON.parse(readFileSync(path.join(dir, "package.json"), "utf8")).scripts ?? {}) as Record<string, string>;
    if (scripts.build) {
      const r = runIn("npm", ["run", "build"]);
      return { compiles: r.code === 0, errors: errLines(r.out).concat(tsErr(r.out)).slice(0, 20), how: "npm run build" };
    }
    const tsconfigs = readdirSync(dir).filter((f) => /^tsconfig.*\.json$/.test(f));
    for (const tc of tsconfigs) {
      const r = runIn("npx", ["tsc", "--noEmit", "-p", tc]);
      if (r.code !== 0) return { compiles: false, errors: tsErr(r.out), how: `tsc -p ${tc}` };
    }
    if (existsSync(path.join(dir, "vite.config.ts")) || existsSync(path.join(dir, "vite.config.js"))) {
      const r = runIn("npx", ["vite", "build"]);
      if (r.code !== 0) return { compiles: false, errors: errLines(r.out).concat(tsErr(r.out)).slice(0, 20), how: "vite build" };
    }
    return { compiles: tsconfigs.length > 0, errors: tsconfigs.length ? [] : ["no build script, no tsconfig — nothing verifiable"], how: `tsc ×${tsconfigs.length}` };
  }
  return { compiles: false, errors: ["no compilable sources delivered at all"], how: "none" };
}

