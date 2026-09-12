import { describe, expect, it } from "vitest";
import { groupSessionsByWorkspace, workspaceName } from "./sessionGroups";

const s = (id: string, workspace: string | null, updatedAt: number) => ({
  id,
  title: id,
  workspace,
  updatedAt,
});

describe("the session rail grouped by workspace", () => {
  it("puts each workspace where its most recent session would be", () => {
    const groups = groupSessionsByWorkspace([
      s("a", "/Users/me/shop", 50),
      s("b", "/Users/me/blog", 40),
      s("c", "/Users/me/shop", 30),
      s("d", null, 20),
    ]);
    expect(groups.map((g) => g.name)).toEqual(["shop", "blog", ""]);
    expect(groups[0].sessions.map((x) => x.id)).toEqual(["a", "c"]);
    expect(groups[2].path).toBeNull();
  });

  it("names a workspace by its folder, on either platform", () => {
    expect(workspaceName("/Users/me/shop/")).toBe("shop");
    expect(workspaceName("C:\\Users\\me\\shop")).toBe("shop");
  });
});
