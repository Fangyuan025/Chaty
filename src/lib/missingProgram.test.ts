import { describe, expect, it } from "vitest";
import { commandPrograms, missingProgram, missingProgramNote } from "./missingProgram";

describe("a program the machine does not have", () => {
  it("is told from every shell's way of saying so", () => {
    expect(missingProgram("bash: python: command not found\n[exit 127]")).toBe("python");
    expect(missingProgram("sh: 1: npx: not found", "npx serve .")).toBe("npx");
    // …but a tool's "x: not found" about something else is not a missing program.
    expect(missingProgram("config.yaml: not found", "mytool --check")).toBeNull();
    expect(missingProgram("zsh: command not found: node")).toBe("node");
    expect(missingProgram("'python' is not recognized as an internal or external command,\noperable program or batch file.")).toBe("python");
    expect(missingProgram("'npx' 不是内部或外部命令，也不是可运行的程序\n或批处理文件。")).toBe("npx");
    expect(missingProgram("The term 'python3' is not recognized as the name of a cmdlet")).toBe("python3");
    expect(missingProgram("无法将“node”项识别为 cmdlet、函数、脚本文件或可运行程序的名称。")).toBe("node");
    expect(missingProgram("Python was not found; run without arguments to install from the Microsoft Store, or disable this shortcut")).toBe("python");
    expect(missingProgram("env: node: No such file or directory")).toBe("node");
    expect(missingProgram("C:\\tools\\Python.EXE: command not found")).toBe("python");
  });

  it("and any other failure is not that", () => {
    expect(missingProgram("Traceback (most recent call last):\n  File \"a.py\", line 1\nNameError: name 'x' is not defined\n[exit 1]")).toBeNull();
    expect(missingProgram("ls: cannot access 'nope': No such file or directory")).toBeNull();
    expect(missingProgram("error TS2304: Cannot find name 'foo'.")).toBeNull();
    expect(missingProgram("npm ERR! 404 Not Found - GET https://registry.npmjs.org/nope")).toBeNull();
    expect(missingProgram("")).toBeNull();
  });
});

describe("the programs a command starts", () => {
  it("the first word of each part, past assignments and wrappers", () => {
    expect(commandPrograms("python -m http.server 8000")).toEqual(["python"]);
    expect(commandPrograms("cd site && PORT=8000 nohup python3 server.py > log 2>&1 &")).toEqual(["cd", "python3"]);
    expect(commandPrograms("[detached] npx serve . | head -5")).toEqual(["npx", "head"]);
    expect(commandPrograms("C:\\Python312\\python.exe app.py; echo done")).toEqual(["python", "echo"]);
    expect(commandPrograms("sudo env FOO=1 node x.js")).toEqual(["node"]);
  });
});

describe("what the model is told", () => {
  it("that it is the machine, not the command — and for a page, that the file opens without it", () => {
    const zh = missingProgramNote("python", true);
    expect(zh).toContain("这台机器上没有 `python`");
    expect(zh).toContain("browser_navigate");
    const en = missingProgramNote("cargo", false);
    expect(en).toContain("no `cargo`");
    expect(en).not.toContain("browser_navigate");
    expect(missingProgramNote("npx", false, true).startsWith("ERROR: intercepted, not run")).toBe(true);
  });
});
