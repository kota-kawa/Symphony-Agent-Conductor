export const $ = <T extends Element = Element>(
  selector: string,
  context: ParentNode = document,
): T | null => context.querySelector(selector) as T | null;

export const $$ = <T extends Element = Element>(
  selector: string,
  context: ParentNode = document,
): T[] => Array.from(context.querySelectorAll(selector)) as T[];

export function escapeHTML(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, (match) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[match] ?? match);
}
