import test from "node:test";
import assert from "node:assert/strict";
import { TuiMainScreen, matchesKey, parseKey, isKittyProtocolActive, setKittyProtocolActive, isKeyRelease, isKeyRepeat, type Terminal } from "@earendil-works/pi-tui";

test("1.2 legacy and Kitty Ctrl-C/ESC and shifted keys remain distinguishable", () => {
    const prior = isKittyProtocolActive();
    try {
        setKittyProtocolActive(true);
        assert.equal(isKeyRelease("\x1b[99;5:3u"), true);
        assert.equal(isKeyRepeat("\x1b[99;5:2u"), true);
        assert.equal(isKeyRelease("\x1b[99;5:1u"), false);
    } finally { setKittyProtocolActive(prior); }
    assert.equal(isKittyProtocolActive(), prior);
    assert.equal(matchesKey("\x03", "ctrl+c"), true);
    assert.equal(matchesKey("\x1b", "escape"), true);
    assert.equal(matchesKey("\x1b[99;5u", "ctrl+c"), true);
    assert.equal(matchesKey("\x1b[27u", "escape"), true);
    assert.equal(matchesKey("\x1b[99;5u", "escape"), false);
    assert.equal(parseKey("\x1b[97;2u"), "shift+a");
    assert.equal(matchesKey("\x1b[97;2u", "shift+a"), true);
});

class OwnedTerminal implements Terminal {
    columns = 80;
    rows = 24;
    kittyProtocolActive = false;
    async drainInput(..._args: Parameters<Terminal["drainInput"]>): Promise<void> {}
    setProgress(..._args: Parameters<Terminal["setProgress"]>): void {}
    setProgramStatus(..._args: Parameters<Terminal["setProgramStatus"]>): void {}
    writes: string[] = [];
    starts = 0;
    stops = 0;
    input?: (data: string) => void;
    start(onInput: (data: string) => void, _onResize: () => void): void { this.starts++; this.input = onInput; }
    stop(): void { this.stops++; this.input = undefined; }
    write(data: string): void { this.writes.push(data); }
    moveBy(_lines: number): void {}
    hideCursor(): void { this.write("\x1b[?25l"); }
    showCursor(): void { this.write("\x1b[?25h"); }
    clearLine(): void {}
    clearFromCursor(): void {}
    clearScreen(): void {}
    setTitle(_title: string): void {}
}

test("genuine 1.2 TUI on owned terminal routes cancel keys and restores cursor at shutdown", () => {
    const terminal = new OwnedTerminal();
    const tui = new TuiMainScreen(terminal);
    assert.equal(tui.mode, "regular", "standalone application uses the main screen, not alternate viewport");
    const seen: string[] = [];
    const component = { render: (_width: number) => ["owned terminal"], invalidate() {},
        handleInput(data: string) { seen.push(data); } };
    tui.addChild(component);
    tui.setFocus(component);
    const originalExit = process.exit;
    try {
        tui.start();
        assert.equal(terminal.starts, 1);
        for (const data of ["\x03", "\x1b", "\x1b[99;5u", "\x1b[27u"]) terminal.input!(data);
        assert.deepEqual(seen, ["\x03", "\x1b", "\x1b[99;5u", "\x1b[27u"]);
        assert.equal(process.exit, originalExit, "TUI must not replace process.exit");
    } finally { tui.stop(); }
    assert.equal(terminal.stops, 1);
    assert.equal(terminal.input, undefined);
    assert.ok(terminal.writes.join("").includes("\x1b[?25h"), "shutdown restores visible cursor");
});
