import { describe, expect, test } from "vitest";
import { isReadOnlyCommand, isSymbolicCheck } from "./readOnlyCmd";

const PASS: string[] = [
  "ls",
  "ls -la",
  "ls -la src",
  "cat README.md",
  "pwd",
  "head -50 src/main.rs",
  "tail -n 20 log.txt",
  "wc -l src/App.tsx",
  "grep -rn pattern src",
  "FOO=1 grep -rn pat src",
  "A=1 B=2 env",
  "rg TODO src",
  "rg -n 'fn main' src",
  "which python3",
  "git status",
  "git log --oneline | head -20",
  "git diff HEAD~1",
  "git show abc123",
  "git blame src/lib.rs",
  "git branch",
  "git branch -a",
  "git tag --list",
  "git stash list",
  "git stash show",
  "git config --list",
  "git config --get user.name",
  "git rev-parse HEAD",
  "cd src && ls",
  "cd src",
  "cd src; ls",
  "find . -name '*.ts' | wc -l",
  "find src -type f -newer a.txt",
  "cargo check",
  "sort file.txt",
  "sort -r file.txt | uniq -c",
  "diff a.txt b.txt",
  "du -sh node_modules",
  "df -h",
  "ps aux | grep node",
  "echo hello",
  "printf '%s' hi",
  "test -f Cargo.toml && cat Cargo.toml",
  "stat src/main.rs",
  "tree -L 2",
  "cat a.txt < b.txt",
];

const REJECT: string[] = [
  "",
  "   ",
  ";ls",
  "ls > out.txt",
  "ls >> out.txt",
  "ls 2> err.txt",
  "cat a &> b",
  "echo `rm -rf .`",
  "echo $(rm -rf .)",
  "diff <(sort a) b",
  "ls; rm x",
  "ls && rm x",
  "ls & rm x",
  "ls | tee out.txt",
  "find . -delete",
  "find . -name '*.o' -exec rm {} +",
  "find . -execdir rm {} +",
  "find . -fprintf log '%p'",
  "rg --pre cat query",
  "rg --pre=cat query",
  "sort -o out.txt in.txt",
  "sort --output=x in",
  "uniq in.txt out.txt",
  "env rm -rf x",
  "env -i sh",
  "git checkout main",
  "git push",
  "git pull",
  "git branch -d topic",
  "git branch --edit-description",
  "git tag v1.0",
  "git stash",
  "git stash pop",
  "git config user.name me",
  "git config --edit --list",
  "git remote add o url",
  "cargo build",
  "cargo run",
  "npm ls",
  "sudo ls",
  "sed -i 's/a/b/' f.txt",
  "./run.sh",
  "/bin/ls",
  "python3 -c 'print(1)'",
];

describe("isReadOnlyCommand", () => {
  for (const cmd of PASS) {
    test(`pass: ${JSON.stringify(cmd)}`, () => {
      expect(isReadOnlyCommand(cmd)).toBe(true);
    });
  }
  for (const cmd of REJECT) {
    test(`reject: ${JSON.stringify(cmd)}`, () => {
      expect(isReadOnlyCommand(cmd)).toBe(false);
    });
  }
  test("windows rejects everything", () => {
    expect(isReadOnlyCommand("dir", { windows: true })).toBe(false);
    expect(isReadOnlyCommand("ls", { windows: true })).toBe(false);
  });
});

describe("isSymbolicCheck", () => {
  const SYMBOLIC = [
    "swift --version",
    "swiftc --version",
    "node -v",
    "python3 --help",
    "cd CalendarApp && swiftc -parse a.swift b.swift",
    "xcrun swiftc -parse Views/App.swift",
    "swiftc -dump-parse main.swift",
    "swift --version && swiftc -parse x.swift",
  ];
  const REAL = [
    "xcodebuild -project X.xcodeproj build",
    "swiftc -typecheck a.swift",
    "python3 tool.py",
    "cargo build",
    "npm test",
    "swift build",
    "ls -la", // read-only, not symbolic — different classifier
    "swiftc -parse a.swift > out.txt", // unparsed shape: fail closed
  ];
  for (const cmd of SYMBOLIC) {
    test(`symbolic: ${JSON.stringify(cmd)}`, () => {
      expect(isSymbolicCheck(cmd)).toBe(true);
    });
  }
  for (const cmd of REAL) {
    test(`real: ${JSON.stringify(cmd)}`, () => {
      expect(isSymbolicCheck(cmd)).toBe(false);
    });
  }
});
