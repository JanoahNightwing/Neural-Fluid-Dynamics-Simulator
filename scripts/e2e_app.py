"""End-to-end check of the running app in a real browser (headless Chrome via Playwright).

    python -m nfds.app --models runs --port 8765 &
    python scripts/e2e_app.py --url http://127.0.0.1:8765 --out figures/app

Loads the page, waits for the first frame, plays, paints a vortex, pushes the viscosity
outside the training range, switches predictor, and saves screenshots along the way.
Fails if the canvases stay blank, the time does not advance, or the page logs errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--out", default="figures/app")
    p.add_argument("--chrome", default="/usr/bin/google-chrome")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []

    with sync_playwright() as pw:
        gl_flags = ["--enable-unsafe-swiftshader", "--use-angle=swiftshader"]  # WebGL headless
        browser = pw.chromium.launch(executable_path=args.chrome, headless=True, args=gl_flags)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.on("console", lambda m: problems.append(f"console {m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.goto(args.url)

        def time_value() -> float:
            return float(page.locator(".readout dd").first.inner_text())

        def canvas_is_drawn(i: int) -> bool:
            # sample the canvas: a drawn vorticity field has many distinct colours
            return page.evaluate("""(i) => {
                const c = document.querySelectorAll('canvas.field')[i];
                const tmp = document.createElement('canvas');
                tmp.width = 32; tmp.height = 32;
                const ctx = tmp.getContext('2d');
                ctx.drawImage(c, 0, 0, 32, 32);
                const d = ctx.getImageData(0, 0, 32, 32).data;
                const seen = new Set();
                for (let k = 0; k < d.length; k += 4) seen.add(d[k] + ',' + d[k+1] + ',' + d[k+2]);
                return seen.size > 20;
            }""", i)

        page.wait_for_function("document.querySelector('.readout dd') !== null")
        page.wait_for_timeout(2500)
        for i in range(2):
            if not canvas_is_drawn(i):
                problems.append(f"canvas {i} is blank after the first frame")
        page.screenshot(path=out / "01_start.png")

        page.get_by_role("button", name="Play").click()
        page.wait_for_timeout(6000)
        t1 = time_value()
        if t1 <= 0:
            problems.append("time did not advance while playing")
        page.screenshot(path=out / "02_playing.png")

        # paint a strong vortex into the middle of the solver panel
        box = page.locator("canvas.field").first.bounding_box()
        cx, cy = box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5
        page.mouse.move(cx, cy)
        page.mouse.down()
        for k in range(8):
            page.mouse.move(cx + 6 * k, cy - 4 * k)
            page.wait_for_timeout(60)
        page.mouse.up()
        page.wait_for_timeout(1500)
        page.screenshot(path=out / "03_painted.png")

        # push the viscosity far outside the training range (most turbulent end)
        slider = page.get_by_label("Reynolds number")
        slider.focus()
        for _ in range(60):
            page.keyboard.press("ArrowRight")
        page.keyboard.press("End")
        page.wait_for_timeout(400)
        note = page.locator(".visc-note").inner_text()
        if "extrapolating" not in note:
            problems.append(f"viscosity note did not warn about extrapolation: {note!r}")
        page.wait_for_timeout(4000)
        page.screenshot(path=out / "04_extrapolating.png")

        page.get_by_label("Predictor").select_option("coarse_solver")
        page.wait_for_timeout(4000)
        page.screenshot(path=out / "05_baseline.png")
        t2 = time_value()
        if t2 <= t1:
            problems.append("time stopped advancing")

        page.set_viewport_size({"width": 420, "height": 900})
        page.wait_for_timeout(800)
        page.screenshot(path=out / "06_mobile.png", full_page=True)
        browser.close()

    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"ok: time advanced to {t2:.2f}; screenshots in {out}")


if __name__ == "__main__":
    main()
