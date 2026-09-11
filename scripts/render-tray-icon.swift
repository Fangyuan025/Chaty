// Render the macOS menu-bar glyph from its SVG source:
//
//   swift scripts/render-tray-icon.swift src-tauri/icons/tray-template.svg src-tauri/icons 36
//   mv src-tauri/icons/tray-36.png src-tauri/icons/tray-template.png
//
// 36 px because tray-icon sets the status item 18 pt tall, which is exactly
// @2x on a Retina menu bar. The glyph is black on transparency: macOS uses only
// its alpha as a TEMPLATE and tints it to match the bar. AppKit's own NSImage
// reads the SVG, so this needs nothing beyond Xcode's Swift.
import AppKit
let args = CommandLine.arguments
let svg = URL(fileURLWithPath: args[1]), outDir = URL(fileURLWithPath: args[2])
guard let img = NSImage(contentsOf: svg) else { print("NSImage could not read the SVG"); exit(1) }
for px in args[3...].compactMap({ Int($0) }) {
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    NSGraphicsContext.current?.imageInterpolation = .high
    img.draw(in: NSRect(x: 0, y: 0, width: px, height: px))
    NSGraphicsContext.restoreGraphicsState()
    let data = rep.representation(using: .png, properties: [:])!
    try! data.write(to: outDir.appendingPathComponent("tray-\(px).png"))
    print("wrote tray-\(px).png")
}
