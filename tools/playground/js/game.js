// A small top-down town drawn on a canvas with CC0 pixel art (assets/LICENSE-assets.txt): ground tiles, a shop, an
// infirmary tower, the town gate, a fountain, trees; the three characters wander their patch and the visitor clicks
// one to talk. No player avatar: the camera is the player.

const TILE = 32;
const SOURCE_TILE = 16;
const SCALE = TILE / SOURCE_TILE;
const COLUMNS = 20;
const ROWS = 14;
const FRAME_W = 16;
const FRAME_H = 32;
const WALK_FRAMES = 4;
const FRAME_TICKS = 9;
const WANDER_SPEED = 0.012;
const WANDER_RADIUS = 1.6;
const PAUSE_FRAMES = [60, 200];
const HIT_RADIUS = 0.9;
const BUBBLE_FRAMES = 240;
const BUBBLE_CHARS = 60;
const LABEL_FONT = "12px system-ui, sans-serif";
const BUBBLE_FONT = "12px system-ui, sans-serif";

const SHEETS = { overworld: "assets/overworld.png", character: "assets/character.png" };

const GROUND = {
  G: { sx: 0, sy: 48 },
  P: { sx: 112, sy: 176 },
  W: { sx: 272, sy: 0 },
};

const MAP = [
  "GGGGGGGGGGGGGGGGGGGG",
  "GGGGGGGGGGGGGGGGGGGG",
  "GGGGGGGGGGGGGGGGGGGG",
  "GGGGGGGGGGGGGGGGGGGG",
  "GGGGGGGGGGGGGGGGGGGG",
  "GGGPPPPPPPPPPPPPPGGG",
  "GGGPGGGGGGGGGGGGPGGG",
  "WWGPGGGGGGGGGGGGPGWW",
  "WWGPPPPPPPPPPPPPPGWW",
  "GGGGGGGGGPPGGGGGGGGG",
  "GGGGGGGGGPPGGGGGGGGG",
  "GGGGGGGGGPPGGGGGGGGG",
  "GGGGGGGGGPPGGGGGGGGG",
  "GGGGGGGGGPPGGGGGGGGG",
];

// Sprites from the overworld sheet: source rectangle in pixels, placement in tiles, and whether they block walking.
const DECORATIONS = [
  { name: "shop", sx: 96, sy: 0, sw: 80, sh: 80, x: 1, y: 0, blocks: true },
  { name: "infirmary", sx: 48, sy: 352, sw: 56, sh: 80, x: 14.5, y: 0, blocks: true },
  { name: "fountain", sx: 352, sy: 144, sw: 48, sh: 48, x: 8.5, y: 5.5, blocks: true },
  { name: "gate", sx: 384, sy: 368, sw: 64, sh: 80, x: 8, y: 9.5, blocks: true },
  { name: "wall-left", sx: 320, sy: 448, sw: 64, sh: 96, x: 4, y: 9, blocks: true },
  { name: "wall-right", sx: 320, sy: 448, sw: 64, sh: 96, x: 12, y: 9, blocks: true },
  { name: "tree", sx: 80, sy: 256, sw: 32, sh: 32, x: 0, y: 11, blocks: true },
  { name: "tree", sx: 80, sy: 256, sw: 32, sh: 32, x: 18, y: 11, blocks: true },
  { name: "tree", sx: 80, sy: 256, sw: 32, sh: 32, x: 7, y: 0.5, blocks: true },
  { name: "tree", sx: 80, sy: 256, sw: 32, sh: 32, x: 11, y: 0.5, blocks: true },
  { name: "tree", sx: 80, sy: 256, sw: 32, sh: 32, x: 1, y: 6, blocks: true },
  { name: "tree", sx: 80, sy: 256, sw: 32, sh: 32, x: 17, y: 6, blocks: true },
];

const HOMES = { mira: { x: 4.5, y: 6.6 }, ferreira: { x: 15.2, y: 6.6 }, holt: { x: 9.5, y: 8.6 } };
const TINTS = { mira: "none", ferreira: "hue-rotate(150deg)", holt: "hue-rotate(220deg) brightness(0.85)" };
const FACING_ROWS = { down: 0, right: 1, up: 2, left: 3 };
const BLOCKED_GROUND = new Set(["W"]);

function randomBetween(low, high) {
  return low + Math.random() * (high - low);
}

async function loadImage(source) {
  const response = await fetch(source);
  if (!response.ok) throw new Error(`could not load ${source}: ${response.status}`);
  return createImageBitmap(await response.blob());
}

export class Game {
  constructor(canvas, npcs) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.context.imageSmoothingEnabled = false;
    this.images = {};
    this.npcs = npcs.map((npc) => ({
      ...npc,
      home: HOMES[npc.id],
      x: HOMES[npc.id].x,
      y: HOMES[npc.id].y,
      target: null,
      pause: Math.floor(randomBetween(...PAUSE_FRAMES)),
      facing: "down",
      step: 0,
      ticks: 0,
      bubble: null,
      bubbleFrames: 0,
    }));
    this.selected = null;
    this.hovered = null;
    this.blocked = this.blockedTiles();
    canvas.width = COLUMNS * TILE;
    canvas.height = ROWS * TILE;
  }

  async load() {
    const entries = await Promise.all(Object.entries(SHEETS).map(async ([key, path]) => [key, await loadImage(path)]));
    this.images = Object.fromEntries(entries);
  }

  blockedTiles() {
    const blocked = new Set();
    for (const item of DECORATIONS) {
      if (!item.blocks) continue;
      const width = item.sw / SOURCE_TILE;
      const height = item.sh / SOURCE_TILE;
      for (let column = Math.floor(item.x); column < item.x + width; column += 1) {
        for (let row = Math.floor(item.y); row < item.y + height; row += 1) blocked.add(`${column},${row}`);
      }
    }
    return blocked;
  }

  walkable(x, y) {
    const column = Math.floor(x);
    const row = Math.floor(y);
    if (column < 0 || row < 0 || column >= COLUMNS || row >= ROWS) return false;
    if (BLOCKED_GROUND.has(MAP[row][column])) return false;
    return !this.blocked.has(`${column},${row}`);
  }

  pickTarget(npc) {
    for (let attempt = 0; attempt < 8; attempt += 1) {
      const x = npc.home.x + randomBetween(-WANDER_RADIUS, WANDER_RADIUS);
      const y = npc.home.y + randomBetween(-WANDER_RADIUS, WANDER_RADIUS);
      if (this.walkable(x, y) && this.walkable(x, y - 0.5)) return { x, y };
    }
    return null;
  }

  wander(npc) {
    if (npc === this.selected) {
      npc.step = 0;
      return;
    }
    if (npc.pause > 0) {
      npc.pause -= 1;
      npc.step = 0;
      return;
    }
    if (!npc.target) {
      npc.target = this.pickTarget(npc);
      if (!npc.target) npc.pause = PAUSE_FRAMES[0];
      return;
    }
    const dx = npc.target.x - npc.x;
    const dy = npc.target.y - npc.y;
    const distance = Math.hypot(dx, dy);
    if (distance < WANDER_SPEED) {
      npc.target = null;
      npc.pause = Math.floor(randomBetween(...PAUSE_FRAMES));
      npc.facing = "down";
      return;
    }
    const nextX = npc.x + (dx / distance) * WANDER_SPEED;
    const nextY = npc.y + (dy / distance) * WANDER_SPEED;
    if (!this.walkable(nextX, nextY)) {
      npc.target = null;
      return;
    }
    npc.x = nextX;
    npc.y = nextY;
    npc.facing = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? "right" : "left") : dy > 0 ? "down" : "up";
    npc.ticks += 1;
    if (npc.ticks % FRAME_TICKS === 0) npc.step = (npc.step + 1) % WALK_FRAMES;
  }

  update() {
    for (const npc of this.npcs) {
      this.wander(npc);
      if (npc.bubbleFrames > 0) npc.bubbleFrames -= 1;
    }
  }

  npcAt(pixelX, pixelY) {
    const x = pixelX / TILE;
    const y = pixelY / TILE;
    return this.npcs.find((npc) => Math.hypot(npc.x - x, npc.y - 0.5 - y) <= HIT_RADIUS) ?? null;
  }

  select(npc) {
    this.selected = this.npcs.find((candidate) => candidate.id === npc.id) ?? null;
    if (this.selected) this.selected.facing = "down";
  }

  hover(npc) {
    this.hovered = npc;
  }

  say(npcId, text) {
    const npc = this.npcs.find((candidate) => candidate.id === npcId);
    if (!npc) return;
    npc.bubble = text.length > BUBBLE_CHARS ? `${text.slice(0, BUBBLE_CHARS).trim()}...` : text;
    npc.bubbleFrames = BUBBLE_FRAMES;
  }

  draw(state) {
    const context = this.context;
    const sheet = this.images.overworld;
    for (let row = 0; row < ROWS; row += 1) {
      for (let column = 0; column < COLUMNS; column += 1) {
        const ground = GROUND[MAP[row][column]];
        if (sheet) {
          context.drawImage(sheet, ground.sx, ground.sy, SOURCE_TILE, SOURCE_TILE, column * TILE, row * TILE, TILE, TILE);
        } else {
          context.fillStyle = MAP[row][column] === "W" ? "#4d7fa3" : MAP[row][column] === "P" ? "#b89a6a" : "#5f8a4f";
          context.fillRect(column * TILE, row * TILE, TILE, TILE);
        }
      }
    }
    if (state.alarm === "ringing") {
      context.fillStyle = "rgba(220, 60, 40, 0.25)";
      context.fillRect(0, 0, COLUMNS * TILE, ROWS * TILE);
    }
    const drawables = [
      ...DECORATIONS.map((item) => ({ kind: "decoration", item, bottom: (item.y + item.sh / SOURCE_TILE) * TILE })),
      ...this.npcs.map((npc) => ({ kind: "npc", npc, bottom: (npc.y + 0.5) * TILE })),
    ].sort((a, b) => a.bottom - b.bottom);
    for (const drawable of drawables) {
      if (drawable.kind === "decoration") this.drawDecoration(drawable.item);
      else this.drawNpc(drawable.npc);
    }
    this.drawGuards(state);
    for (const npc of this.npcs) this.drawLabel(npc);
    for (const npc of this.npcs) if (npc.bubbleFrames > 0) this.drawBubble(npc);
  }

  drawDecoration(item) {
    const sheet = this.images.overworld;
    if (!sheet) return;
    this.context.drawImage(sheet, item.sx, item.sy, item.sw, item.sh, item.x * TILE, item.y * TILE, item.sw * SCALE, item.sh * SCALE);
  }

  drawNpc(npc) {
    const context = this.context;
    const sheet = this.images.character;
    const x = npc.x * TILE;
    const y = npc.y * TILE;
    const selected = npc === this.selected;
    const hovered = npc === this.hovered;
    if (selected || hovered) {
      context.fillStyle = selected ? "rgba(255, 215, 120, 0.55)" : "rgba(255, 255, 255, 0.35)";
      context.beginPath();
      context.ellipse(x, y + TILE * 0.45, TILE * 0.55, TILE * 0.22, 0, 0, Math.PI * 2);
      context.fill();
    }
    if (!sheet) {
      context.fillStyle = "#e8e1d4";
      context.beginPath();
      context.arc(x, y, 13, 0, Math.PI * 2);
      context.fill();
      return;
    }
    const frame = npc.step % WALK_FRAMES;
    const row = FACING_ROWS[npc.facing] ?? 0;
    context.save();
    context.filter = TINTS[npc.id] ?? "none";
    context.drawImage(
      sheet,
      frame * FRAME_W,
      row * FRAME_H,
      FRAME_W,
      FRAME_H,
      x - (FRAME_W * SCALE) / 2,
      y - FRAME_H * SCALE + TILE * 0.5,
      FRAME_W * SCALE,
      FRAME_H * SCALE,
    );
    context.restore();
  }

  drawLabel(npc) {
    const context = this.context;
    context.font = LABEL_FONT;
    context.textAlign = "center";
    context.textBaseline = "middle";
    const x = npc.x * TILE;
    const y = npc.y * TILE - FRAME_H * SCALE + TILE * 0.5 - 10;
    const width = context.measureText(npc.npc).width + 10;
    context.fillStyle = "rgba(20, 16, 10, 0.6)";
    context.fillRect(x - width / 2, y - 8, width, 16);
    context.fillStyle = "#fff";
    context.fillText(npc.npc, x, y);
  }

  drawBubble(npc) {
    const context = this.context;
    context.font = BUBBLE_FONT;
    const width = Math.min(context.measureText(npc.bubble).width + 16, COLUMNS * TILE - 8);
    const x = Math.min(Math.max(npc.x * TILE - width / 2, 4), COLUMNS * TILE - width - 4);
    const y = npc.y * TILE + TILE * 0.6;
    context.fillStyle = "rgba(255, 253, 248, 0.95)";
    context.fillRect(x, y, width, 22);
    context.strokeStyle = "#4a3b2a";
    context.lineWidth = 1;
    context.strokeRect(x, y, width, 22);
    context.fillStyle = "#2a221a";
    context.textAlign = "left";
    context.textBaseline = "middle";
    context.fillText(npc.bubble, x + 8, y + 11);
  }

  drawGuards(state) {
    if (state.guards !== "arriving") return;
    const sheet = this.images.character;
    const anchor = this.selected ?? this.npcs[0];
    const context = this.context;
    context.save();
    context.filter = "hue-rotate(200deg) saturate(0.4) brightness(0.8)";
    for (const offset of [-1.3, 1.3]) {
      const x = (anchor.x + offset) * TILE;
      const y = (anchor.y + 0.9) * TILE;
      if (sheet) {
        context.drawImage(sheet, 0, FACING_ROWS.up * FRAME_H, FRAME_W, FRAME_H, x - (FRAME_W * SCALE) / 2, y - FRAME_H * SCALE + TILE * 0.5, FRAME_W * SCALE, FRAME_H * SCALE);
      }
    }
    context.restore();
  }
}
