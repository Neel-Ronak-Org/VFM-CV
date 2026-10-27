// Course presentation (CV mini project). Every number comes from presentation/numbers.json, written by
// presentation/make_numbers.py from results/*.csv, except the per-image scores on the demo slides, which are
// quoted from the app's own screenshots. Images come from figures/ and presentation/screens/.
//
//   python presentation/make_numbers.py
//   node presentation/build_deck.js        # -> presentation/CV_mini_project_I015_I032.pptx
//
// Demo screenshots (each slide added when its files are present): presentation/screens/demo_input_{upload,controls}.png,
// demo_output_rain.png (+ assets/demo_corrupted_crop.png), demo_output_gt_{spider,branch}.png;
// intermediate outputs: figures/intermediates/<image>_<corruption><severity>/ from scripts/intermediates.py.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const REPO = path.resolve(__dirname, "..");
const N = JSON.parse(fs.readFileSync(path.join(__dirname, "numbers.json"), "utf8"));
const OUT = path.join(__dirname, "CV_mini_project_I015_I032.pptx");
const APPLY_THEME = process.env.PPTX_APPLY_THEME; // path to the pptx skill's scripts/apply_theme.js (optional)
const fig = (f) => path.join(REPO, "figures", f);
const screen = (f) => path.join(__dirname, "screens", f);
const exists = (p) => fs.existsSync(p);

const COURSE = "Computer Vision";
const MEMBERS = [["Ronak Daniel", "I015"], ["Neel Jaiswal", "I032"]];
const FOOTER = "CV Mini Project · Ronak Daniel (I015) · Neel Jaiswal (I032)";

const THEME = {
  name: "Camouflage",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A21", lt1: "FFFFFF", dk2: "2C4A36", lt2: "EEF3EA",
    accent1: "4F7A3A", accent2: "D9532B", accent3: "A9C77C", accent4: "8C6D46",
    accent5: "5E7F99", accent6: "8A8F8A", hlink: "4F7A3A", folHlink: "8C6D46",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "Robust Camouflaged Object Detection on a Laptop GPU";
pres.author = "Ronak Daniel, Neel Jaiswal";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const W = 13.333, M = 0.6, CW = W - 2 * M;
const fmt3 = (x) => (x === null || x === undefined ? "—" : x.toFixed(3));
const signed = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
const ci = (s) => `95% CI ${signed(s.lo)} to ${signed(s.hi)}`;
const model = (k) => N.models.find((m) => m.key === k);
const OURS = model("Ours-Full"), FIVE = model("Ours-Full-512"), SINET = model("SINet-V2"), S2U = model("SAM2-UNet");
const S = N.significance;
const nTest = ["CHAMELEON", "CAMO", "COD10K", "NC4K"].reduce((a, d) => a + N.images[d], 0);
const mParams = (n) => (n / 1e6).toFixed(1);

async function icon(Comp, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function imgSize(file) {
  // PNG/JPEG pixel size without extra deps: sharp metadata is async, so callers await sizeOf
  return sharp(file).metadata().then((m) => ({ w: m.width, h: m.height }));
}
async function fitImage(slide, file, x, y, maxW, maxH, opts = {}) {
  const { w, h } = await imgSize(file);
  const s = Math.min(maxW / w, maxH / h);
  const iw = w * s, ih = h * s;
  const ix = opts.align === "left" ? x : x + (maxW - iw) / 2;
  slide.addImage({ path: file, x: ix, y, w: iw, h: ih, altText: opts.alt || path.basename(file), objectName: opts.name });
  return { x: ix, y, w: iw, h: ih };
}

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.0, w: 7.0, h: 1.9, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: CW, h: 0.8, fontFace: THEME.headFontFace, fontSize: 34, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOTER, options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

function contentSlide(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function card(slide, x, y, w, h, name, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name });
}
function stat(slide, x, y, w, big, label, name, color = C.accent2) {
  card(slide, x, y, w, 1.45, name + " card");
  slide.addText(big, { x: x + 0.25, y: y + 0.12, w: w - 0.5, h: 0.75, fontFace: THEME.headFontFace, fontSize: 32, bold: true, color, margin: 0, isTextBox: true, objectName: name + " value" });
  slide.addText(label, { x: x + 0.25, y: y + 0.85, w: w - 0.5, h: 0.5, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: name + " label" });
}

async function build() {
  const ic = {
    eye: await icon(fa.FaEye, HEX.lt1), camera: await icon(fa.FaCamera, HEX.lt1), question: await icon(fa.FaQuestion, HEX.lt1),
    db: await icon(fa.FaDatabase, HEX.lt1), bolt: await icon(fa.FaCloudRain, HEX.lt1), cubes: await icon(fa.FaCubes, HEX.lt1),
    brain: await icon(fa.FaBrain, HEX.lt1), chart: await icon(fa.FaChartBar, HEX.lt1), desktop: await icon(fa.FaDesktop, HEX.lt1),
  };

  // 1 ------------------------------------------------------------ title
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText(("Course: " + COURSE).toUpperCase(), { x: M, y: 1.3, w: 7, h: 0.4, fontSize: 14, bold: true, charSpacing: 2, color: C.accent3, margin: 0, isTextBox: true, objectName: "course" });
  s.addText("Robust Camouflaged Object Detection on a Laptop GPU", { placeholder: "title" });
  s.addText("A corruption benchmark of specialist and SAM-based detectors, and a lightweight robust SAM adapter",
    { x: M, y: 3.95, w: 6.8, h: 0.9, fontSize: 18, color: C.background2, margin: 0, valign: "top", isTextBox: true, objectName: "subtitle" });
  MEMBERS.forEach(([name, roll], i) => {
    const x = M + i * 3.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.35, w: 2.85, h: 1.05, rectRadius: 0.08, fill: { color: C.text2 }, line: { color: C.text2 }, objectName: `member ${i + 1} card` });
    s.addText([{ text: name, options: { fontSize: 18, bold: true, color: C.background1, breakLine: true } },
      { text: "Roll no. " + roll, options: { fontSize: 14, color: C.accent3 } }],
      { x: x + 0.25, y: 5.45, w: 2.4, h: 0.85, margin: 0, valign: "middle", isTextBox: true, objectName: `member ${i + 1}` });
  });
  const tiles = [["hero_clean.png", "clean"], ["hero_fog5.png", "fog, severity 5"], ["hero_rain5.png", "rain, severity 5"]];
  tiles.forEach(([f, label], i) => {
    const y = 0.75 + i * 2.05;
    s.addImage({ path: path.join(__dirname, "assets", f), x: 8.45, y, w: 4.25, h: 1.7, sizing: { type: "cover", w: 4.25, h: 1.7 }, altText: `camouflaged flatfish, ${label}`, objectName: `hero ${label}` });
    s.addText(label, { x: 8.45, y: y + 1.72, w: 4.25, h: 0.28, fontSize: 11, color: C.accent3, margin: 0, isTextBox: true, objectName: `hero label ${i + 1}` });
  });
  s.addNotes("Introduce the project and the team. The pictures on the right show the same camouflaged flatfish (a COD10K test image) clean, in fog and in rain: our detector has to find the animal in all three.");

  // 2 ------------------------------------------------------------ problem
  s = contentSlide("The problem: hidden animals, imperfect cameras", "Introduction");
  const rows = [
    [ic.eye, "Camouflaged object detection (COD)", "Find the animal whose colour and texture match its background, and output a pixel mask of it."],
    [ic.camera, "Real cameras degrade images", "Camera traps, drones and phones add blur, noise, JPEG compression, fog and rain."],
    [ic.question, "Open question", "SAM-based detectors are the most accurate on clean images. How robust are they to these corruptions, and can a small model stay robust on a 4 GB laptop GPU?"],
  ];
  rows.forEach(([img, head, body], i) => {
    const y = 1.55 + i * 1.75;
    s.addShape(pres.shapes.OVAL, { x: M, y, w: 0.8, h: 0.8, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `problem icon bg ${i + 1}` });
    s.addImage({ data: img, x: M + 0.2, y: y + 0.2, w: 0.4, h: 0.4, altText: head, objectName: `problem icon ${i + 1}` });
    s.addText(head, { x: M + 1.1, y: y - 0.05, w: 6.2, h: 0.45, fontSize: 20, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: `problem head ${i + 1}` });
    s.addText(body, { x: M + 1.1, y: y + 0.42, w: 6.2, h: 1.0, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `problem body ${i + 1}` });
  });
  stat(s, 8.55, 1.55, 4.18, "8 × 5", "corruption types × severity levels", "stat corruptions");
  stat(s, 8.55, 3.3, 4.18, nTest.toLocaleString("en-US"), "test images in 4 datasets", "stat images");
  stat(s, 8.55, 5.05, 4.18, "4 GB", "laptop GPU (" + N.gpu.replace("NVIDIA GeForce ", "") + ")", "stat gpu");
  s.addNotes("COD is segmentation, not boxes. Cameras in the field rarely give clean images. Nobody had measured how SAM-based COD models behave under corruption, and all the strongest models are too large for a student laptop. These two gaps define the project.");

  // 3 ------------------------------------------------------------ pipeline
  pres.addSection({ title: "Method" });
  s = contentSlide("Implementation pipeline", "Method");
  const steps = [
    [ic.db, "1  Data", `COD10K + CAMO train (${N.train_images.toLocaleString("en-US")} images). Test: CHAMELEON, CAMO, COD10K, NC4K`],
    [ic.bolt, "2  Corruptions", "8 types × 5 severities with fixed seeds. 4 seen in training, 4 held out (unseen)"],
    [ic.cubes, "3  Baselines", "6 published models ported to one interface: predict(image) → mask"],
    [ic.brain, "4  Our model", "Frozen MobileSAM encoder + adapters + learned prompts, trained on the laptop GPU"],
    [ic.chart, "5  Evaluation", "Every model × dataset × 41 conditions (clean + 40). Sα, Eφ, Fβw, MAE, Dice, IoU"],
    [ic.desktop, "6  Analysis + demo", "Retention, unseen corruptions, bootstrap tests, speed and memory. Gradio demo"],
  ];
  const bw = 3.6, bh = 2.25, gx = (CW - 3 * bw) / 2;
  steps.forEach(([img, head, body], i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * (bw + gx), y = 1.45 + row * (bh + 0.45);
    card(s, x, y, bw, bh, `step ${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: i === 3 ? C.accent2 : C.accent1 }, line: { color: i === 3 ? C.accent2 : C.accent1 }, objectName: `step icon bg ${i + 1}` });
    s.addImage({ data: img, x: x + 0.4, y: y + 0.4, w: 0.3, h: 0.3, altText: head, objectName: `step icon ${i + 1}` });
    s.addText(head, { x: x + 1.0, y: y + 0.25, w: bw - 1.2, h: 0.6, fontSize: 19, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `step head ${i + 1}` });
    s.addText(body, { x: x + 0.25, y: y + 1.0, w: bw - 0.5, h: bh - 1.15, fontSize: 14, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `step body ${i + 1}` });
    if (col < 2) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.1, y: y + bh / 2 - 0.15, w: gx - 0.2, h: 0.3, fill: { color: C.accent6 }, line: { color: C.accent6 }, objectName: `arrow ${i + 1}` });
  });
  s.addText("Python 3.11 · PyTorch 2.6 (CUDA 12.4) · PySODMetrics · Gradio · every number read from results/*.csv",
    { x: M, y: 6.45, w: CW, h: 0.35, fontSize: 14, color: C.accent6, margin: 0, isTextBox: true, objectName: "tools" });
  s.addNotes("Walk through the six stages. The rows read left to right; step 4 (orange) is our own model. Stress that all models go through one common interface, so they are compared under identical conditions, and that corruptions use fixed seeds, so every model sees exactly the same corrupted images.");

  // 4 ------------------------------------------------------------ models and algorithms
  s = contentSlide("Models and algorithms used", "Method");
  const groups = [
    ["Specialist COD models", C.accent5, ["SINet-V2 (TPAMI 2022)", "PFNet (CVPR 2021)", "ZoomNet (CVPR 2022)", "FEDER (CVPR 2023, clean only)"]],
    ["SAM-based COD models", C.accent4, [`SAM2-UNet (${model("SAM2-UNet").params_M} M)`, `SAM-Adapter, ViT-H (${model("SAM-Adapter").params_M} M)`, "Oracle-box MobileSAM (analysis only)"]],
    ["Ours: robust SAM adapter", C.accent2, ["MobileSAM TinyViT encoder (frozen)", "Bottleneck adapters in 10 blocks", "Learned prompt tokens", "SAM mask decoder (trained)"]],
  ];
  const gw = (CW - 2 * 0.3) / 3;
  groups.forEach(([head, color, items], i) => {
    const x = M + i * (gw + 0.3);
    card(s, x, 1.4, gw, 2.75, `model group ${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.6, w: 0.32, h: 0.32, fill: { color }, line: { color }, objectName: `model group dot ${i + 1}` });
    s.addText(head, { x: x + 0.7, y: 1.55, w: gw - 0.9, h: 0.42, fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `model group head ${i + 1}` });
    s.addText(items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
      { x: x + 0.25, y: 2.1, w: gw - 0.45, h: 1.95, fontSize: 15, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 4, isTextBox: true, objectName: `model group items ${i + 1}` });
  });
  s.addText("Algorithms", { x: M, y: 4.4, w: CW, h: 0.4, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "algorithms head" });
  const algos = ["Bottleneck adapters", "Learned prompt tokens", "Structure loss (wBCE + wIoU)", "Corruption augmentation",
    "Clean/corrupted consistency loss", "AdamW + cosine learning rate", "fp16 + gradient checkpointing", "Paired bootstrap (95% CI)"];
  const aw = (CW - 3 * 0.25) / 4;
  algos.forEach((t, i) => {
    const x = M + (i % 4) * (aw + 0.25), y = 4.9 + Math.floor(i / 4) * 0.95;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: aw, h: 0.75, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.25 }, objectName: `algo ${i + 1}` });
    s.addText(t, { x: x + 0.15, y, w: aw - 0.3, h: 0.75, fontSize: 14, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `algo text ${i + 1}` });
  });
  s.addNotes("Three families of models: CNN specialists designed for COD, large SAM-based detectors, and our small SAM-based adapter. FEDER released only prediction maps, so it appears in the clean results only. The oracle-box MobileSAM is given the ground-truth box, so it is an analysis probe, not a competitor. The algorithm row lists the training and evaluation techniques; the next two slides explain their roles.");

  // 5 ------------------------------------------------------------ architecture
  s = contentSlide("Our model: a prompt-free MobileSAM adapter", "Method");
  await fitImage(s, fig("fig_architecture.png"), M, 1.35, CW, 3.85, { name: "architecture figure", alt: "Architecture: input, frozen TinyViT encoder with adapters, SAM mask decoder with learned prompts, camouflage map" });
  stat(s, M, 5.35, 3.85, mParams(N.params.total) + " M", "parameters in total", "stat total", C.text1);
  stat(s, M + 4.15, 5.35, 3.85, mParams(N.params.trainable) + " M", "trained: adapters, prompts, decoder", "stat trainable");
  stat(s, M + 8.3, 5.35, 3.83, "0 clicks", "prompts are learned, fully automatic", "stat prompts", C.accent1);
  s.addNotes("The image goes through MobileSAM's TinyViT encoder, which stays frozen. Small bottleneck adapters, added beside each block's MLP, learn what camouflage looks like. SAM normally needs a click or a box; we replace those with learned prompt tokens, so the detector needs no user input. The SAM mask decoder is fine-tuned and outputs the camouflage map. During training each image is shown clean and corrupted; the consistency loss pushes the two predictions to agree.");

  // 6 ------------------------------------------------------------ roles
  s = contentSlide("Role of models and algorithms used", "Method");
  const roles = [
    ["SINet-V2, PFNet, ZoomNet", "Specialist baselines: how well do CNNs built for COD survive corruption? SINet-V2 is the reference for mCE."],
    ["SAM2-UNet, SAM-Adapter", "Foundation-model baselines: the most accurate on clean images; we measure their robustness for the first time."],
    ["MobileSAM TinyViT (frozen)", "General visual features learnt from 11 M images, small enough for a 4 GB GPU."],
    ["Bottleneck adapters", `Teach the frozen encoder about camouflage with ${mParams(201616)} M parameters; start at zero, so training begins from SAM.`],
    ["Learned prompt tokens", "Replace user clicks and boxes, so detection is fully automatic."],
    ["Structure loss", "Weighted BCE + weighted IoU: focuses training on hard pixels near object boundaries."],
    ["Augmentation + consistency", "Train on 4 seen corruptions; make the corrupted-view prediction match the clean one."],
    ["Metrics + bootstrap", "Six standard COD metrics; 95% confidence intervals show whether a difference is real."],
  ];
  const head = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 15 } });
  s.addTable([[head("Component"), head("Role in the project")],
    ...roles.map(([a, b], i) => [
      { text: a, options: { bold: true, color: HEX.dk1, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } },
      { text: b, options: { color: HEX.dk2, fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } }])],
  { x: M, y: 1.35, w: CW, colW: [3.4, CW - 3.4], fontSize: 14, fontFace: THEME.bodyFontFace, valign: "middle", rowH: 0.6, border: { type: "none" }, margin: [0, 0.12, 0, 0.12], objectName: "roles table" });
  s.addNotes("One line per component. The baselines answer the benchmark question; the frozen encoder, adapters and prompts make the model small and automatic; augmentation and the consistency loss target robustness; the bootstrap tells us which differences are statistically real.");

  // 7 ------------------------------------------------------------ inputs
  pres.addSection({ title: "Results" });
  s = contentSlide("Inputs: test images under eight corruptions", "Results");
  const box = await fitImage(s, fig("corruption_preview.jpg"), M, 1.3, 6.9, 5.55, { align: "left", name: "corruption grid", alt: "One COD10K image under 8 corruptions at severities 0 to 5" });
  const rx = box.x + box.w + 0.45, rw = W - M - rx;
  s.addText("Test sets", { x: rx, y: 1.35, w: rw, h: 0.4, fontSize: 18, bold: true, color: C.text1, margin: 0, isTextBox: true, objectName: "test sets head" });
  s.addTable(["CHAMELEON", "CAMO", "COD10K", "NC4K"].map((d, i) => [
    { text: d, options: { fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } },
    { text: N.images[d].toLocaleString("en-US") + " images", options: { align: "right", fill: { color: i % 2 ? HEX.lt1 : HEX.lt2 } } }]),
  { x: rx, y: 1.85, w: rw, colW: [rw * 0.5, rw * 0.5], fontSize: 14, color: HEX.dk2, rowH: 0.4, border: { type: "none" }, margin: [0, 0.1, 0, 0.1], objectName: "test sets table" });
  s.addText([
    { text: "Seen in training", options: { bold: true, color: HEX.accent1, breakLine: true } },
    { text: "Gaussian noise, Gaussian blur, brightness, JPEG", options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Held out (unseen)", options: { bold: true, color: HEX.accent2, breakLine: true } },
    { text: "motion blur, contrast, fog, rain" }],
  { x: rx, y: 3.65, w: rw, h: 1.6, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "seen unseen" });
  s.addText("Severity 1 to 5 follows the COD10K-C definitions; rain is our own addition.",
    { x: rx, y: 5.45, w: rw, h: 0.8, fontSize: 14, italic: true, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "severity note" });
  s.addNotes("Each test image is evaluated clean and under 8 corruptions at 5 severities: 41 versions per image. Four corruptions are used during training (seen); the other four, including rain, are never seen in training, which tests real generalisation rather than memorised augmentation.");

  if (exists(screen("demo_input_upload.png")) && exists(screen("demo_input_controls.png"))) {
    s = contentSlide("Inputs in the demo app", "Results");
    const u = await fitImage(s, screen("demo_input_upload.png"), M, 1.4, 7.95, 3.6, { align: "left", name: "demo upload screenshot", alt: "Gradio demo: photo upload and gallery of test images" });
    const k = await fitImage(s, screen("demo_input_controls.png"), u.x + u.w + 0.3, 1.4, W - M - (u.x + u.w + 0.3), 3.6, { align: "left", name: "demo controls screenshot", alt: "Gradio demo: model choice, corruption and severity controls" });
    s.addText("Upload any photo, or pick one of 12 test images (COD10K, CAMO, CHAMELEON) or a photo from demo/my_photos.",
      { x: u.x, y: 5.2, w: u.w, h: 0.9, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "demo upload caption" });
    s.addText("Tick up to 6 models, choose a corruption (UNSEEN ones were never used in training) and a severity from 0 to 5.",
      { x: k.x, y: 5.2, w: W - M - k.x, h: 0.9, fontSize: 15, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "demo controls caption" });
    s.addNotes("Run with: python demo/app.py. Left: the photo input and the test-image gallery (the pineapple photo is a CAMO test image). Right: the controls. Every model runs on the same corrupted image, so the comparison is fair.");
  }

  // 8 ------------------------------------------------------------ intermediate outputs
  s = contentSlide("Intermediate outputs of our model", "Results");
  const interDir = path.join(REPO, "figures", "intermediates");
  const inter = exists(interDir) ? fs.readdirSync(interDir).filter((d) => exists(path.join(interDir, d, "panel.png"))).sort() : [];
  if (inter.length) {
    const ip = await fitImage(s, path.join(interDir, inter[0], "panel.png"), M, 1.25, CW, 5.2, { name: "intermediate panel", alt: "Eight stages from input to detection" });
    s.addText("Ours (512) on the slide-10 katydid under motion blur, severity 4 (unseen in training). Stages 3–5 are inside the network; sizes are height × width.",
      { x: M, y: ip.y + ip.h + 0.08, w: CW, h: 0.4, fontSize: 12, color: C.accent6, margin: 0, isTextBox: true, objectName: "intermediate caption" });
  } else {
    card(s, M, 1.4, CW, 5.3, "intermediate placeholder", C.background2);
    s.addText("Run on the laptop:  python scripts/intermediates.py\nthen rebuild the deck (figures/intermediates/…/panel.png is picked up automatically).",
      { x: M + 0.5, y: 3.4, w: CW - 1, h: 1.2, fontSize: 18, color: C.text2, align: "center", valign: "middle", isTextBox: true, objectName: "intermediate todo" });
  }
  s.addNotes("Stages inside our model for one test image: the corrupted input, the square network input, the encoder embedding (256 channels on a coarse grid, shown with its first three principal components as colours), the decoder's low-resolution mask, and the final map resized to the original image, then thresholded into the overlay.");

  // 9 ------------------------------------------------------------ outputs grid
  s = contentSlide("Outputs: predicted masks under corruption", "Results");
  const g = await fitImage(s, fig("fig_qualitative.png"), M, 1.25, CW, 5.2, { name: "qualitative grid", alt: "Masks of six models on five corrupted COD10K images" });
  s.addText("Rows 1–3: images where Ours (512) has its median Sα. Rows 4–5: its lowest-scoring images (failure cases). Numbers: per-image Sα.",
    { x: M, y: g.y + g.h + 0.08, w: CW, h: 0.4, fontSize: 12, color: C.accent6, margin: 0, isTextBox: true, objectName: "grid caption" });
  s.addNotes("Each row is one corrupted COD10K image; columns are the input, the ground truth and six models. Images were chosen by fixed rules (median and worst case for our 512 model), not hand-picked; the selection is logged in results/qualitative_selection.csv. The last two rows show where our model fails: heavy motion blur and heavy rain.");

  // 10 ----------------------------------------------------------- outputs teaser
  s = contentSlide("Outputs: one image under increasing motion blur", "Results");
  const t = await fitImage(s, fig("fig_teaser.png"), M, 1.3, 5.6, 5.5, { align: "left", name: "teaser", alt: "Katydid under motion blur: SINet-V2 vs Ours (512)" });
  const tx = t.x + t.w + 0.5, tw = W - M - tx, T = N.teaser;
  s.addText([
    { text: "Motion blur is an unseen corruption: our model never saw it in training.", options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: `SINet-V2 finds only part of the katydid at every severity (Sα ${fmt3(T["SINet-V2_0"])} → ${fmt3(T["SINet-V2_4"])}).`, options: { breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: `Ours (512) recovers its legs and wings (Sα ${fmt3(T["Ours-Full-512_0"])} clean, ${fmt3(T["Ours-Full-512_4"])} at severity 4).` },
  ], { x: tx, y: 1.45, w: tw, h: 3.2, fontSize: 16, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "teaser text" });
  card(s, tx, 4.95, tw, 1.55, "teaser note card");
  s.addText("Chosen by a fixed rule: the image at the 90th percentile of our gain over SINet-V2 under motion blur, a clear example but not the best one.",
    { x: tx + 0.25, y: 5.05, w: tw - 0.5, h: 1.35, fontSize: 14, italic: true, color: C.text2, margin: 0, valign: "middle", isTextBox: true, objectName: "teaser note" });
  s.addNotes("Top row: input; middle: SINet-V2; bottom: our 512-pixel model. Our model's score drops as blur grows, but it keeps far more of the object than SINet-V2. Say clearly that this example was picked by a rule and is better than average.");

  if (exists(screen("demo_output_rain.png"))) {
    s = contentSlide("Outputs in the demo app: rain, severity 4", "Results");
    const crop = path.join(__dirname, "assets", "demo_corrupted_crop.png");
    const ci0 = await fitImage(s, crop, M, 1.35, 4.5, 4.6, { align: "left", name: "demo corrupted input", alt: "Pineapple photo with rain at severity 4, as fed to every model" });
    s.addText("Input after the corruption (rain is unseen in training)", { x: ci0.x, y: ci0.y + ci0.h + 0.08, w: ci0.w, h: 0.6, fontSize: 12, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "demo input label" });
    const ox = ci0.x + ci0.w + 0.35;
    const o = await fitImage(s, screen("demo_output_rain.png"), ox, 1.35, W - M - ox, 3.2, { align: "left", name: "demo rain outputs", alt: "SINet-V2, SAM2-UNet and Ours-Full overlays with latency" });
    s.addText([
      { text: "A CAMO test image: a face painted to look like a pineapple.", options: { bullet: true, breakLine: true } },
      { text: "SINet-V2 marks almost every pineapple; SAM2-UNet and Ours-Full pick out the painted face.", options: { bullet: true, breakLine: true } },
      { text: "Latency shown by the app: SINet-V2 114 ms, SAM2-UNet 166 ms, Ours-Full 51 ms. It includes pre- and post-processing, so it is higher than the benchmark timing.", options: { bullet: true } },
    ], { x: ox, y: o.y + o.h + 0.25, w: W - M - ox, h: 6.7 - (o.y + o.h + 0.25), fontSize: 15, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true, objectName: "demo rain text" });
    s.addNotes("Each selected model's mask is drawn as a coloured overlay on the corrupted image, with its latency underneath.");
  }

  if (exists(screen("demo_output_gt_spider.png")) && exists(screen("demo_output_gt_branch.png"))) {
    s = contentSlide("Outputs in the demo app, with ground truth", "Results");
    const a = await fitImage(s, screen("demo_output_gt_spider.png"), M, 1.3, 6.4, 2.75, { align: "left", name: "demo spider", alt: "Spider under rain: SINet-V2, SAM2-UNet and Ours-Full with per-image scores" });
    const b = await fitImage(s, screen("demo_output_gt_branch.png"), M, a.y + a.h + 0.2, 6.4, 2.75, { align: "left", name: "demo branch", alt: "Animal on a branch under rain: three models with per-image scores" });
    const gx2 = Math.max(a.x + a.w, b.x + b.w) + 0.45, gw2 = W - M - gx2;
    card(s, gx2, 1.3, gw2, 2.55, "demo spider card");
    s.addText([{ text: "Top: a spider in dry grass", options: { bold: true, color: HEX.dk1, breakLine: true } },
      { text: "Ours-Full scores highest on this image (S-measure 0.851, against 0.790 for SAM2-UNet and 0.683 for SINet-V2)." }],
    { x: gx2 + 0.25, y: 1.45, w: gw2 - 0.5, h: 2.25, fontSize: 17, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "demo spider text" });
    card(s, gx2, 4.05, gw2, 2.55, "demo branch card");
    s.addText([{ text: "Bottom: an animal on a branch", options: { bold: true, color: HEX.dk1, breakLine: true } },
      { text: "Here SINet-V2 is best (0.878; Ours-Full 0.803, SAM2-UNet 0.759). Single images go either way; the benchmark averages thousands." }],
    { x: gx2 + 0.25, y: 4.2, w: gw2 - 0.5, h: 2.25, fontSize: 17, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: "demo branch text" });
    s.addNotes("With 'Show ground truth' ticked for a test image, the app draws the ground truth in white and prints each model's per-image S-measure and MAE. Scores on this slide are the ones printed by the app. We show one image where our model wins and one where it does not, to be honest about single examples.");
  }

  // 11 ----------------------------------------------------------- performance: accuracy
  pres.addSection({ title: "Performance" });
  s = contentSlide("Performance: accuracy under corruption (COD10K)", "Performance");
  s.addChart(pres.charts.BAR, [
    { name: "Clean", labels: N.models.map((m) => m.name), values: N.models.map((m) => m.clean) },
    { name: "Under corruption (mean of 40)", labels: N.models.map((m) => m.name), values: N.models.map((m) => m.mcorr) },
  ], {
    x: M, y: 1.3, w: 8.0, h: 5.5, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.accent3, HEX.accent1], valAxisMinVal: 0.7, valAxisMaxVal: 0.9, valAxisMajorUnit: 0.05,
    valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000",
    dataLabelFontSize: 9, dataLabelColor: HEX.dk2, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.accent6, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "DDDDDD", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "t", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.dk2,
    showTitle: true, title: "Structure measure Sα (higher is better)", titleFontSize: 14, titleColor: HEX.dk1, titleFontFace: "+mn-lt",
    objectName: "accuracy chart",
  });
  const sx = 9.0, sw = W - M - sx;
  stat(s, sx, 1.4, sw, signed(S.ours_sinet.diff), `Ours (1024) vs SINet-V2 under corruption (${ci(S.ours_sinet)})`, "stat vs sinet");
  stat(s, sx, 3.2, sw, signed(S.ours_sam2unet.diff), `Ours (1024) vs SAM2-UNet (${model("SAM2-UNet").params_M} M params): still behind`, "stat vs sam2unet", C.accent5);
  stat(s, sx, 5.0, sw, signed(S.full_clean_unseen.diff), "on unseen corruptions, from augmentation + consistency loss", "stat unseen", C.accent1);
  s.addNotes(`Light bars are clean scores, dark bars the mean over all 40 corrupted conditions. Ours (1024) beats every specialist model under corruption on COD10K, and the paired bootstrap confirms it (+${S.ours_sinet.diff.toFixed(3)} over SINet-V2). It stays below SAM2-UNet, which is 20 times larger; we report that honestly. Training on four corruptions also helps on the four unseen ones, by a small but significant amount (${ci(S.full_clean_unseen)}).`);

  // 12 ----------------------------------------------------------- performance: efficiency
  s = contentSlide("Performance: speed and memory on the laptop GPU", "Performance");
  const hdr = ["Model", "Params (M)", "Latency (ms)", "FPS", "Peak VRAM (MB)", "Sα under corruption"].map((x) => head(x));
  const isOurs = (m) => m.key.startsWith("Ours");
  s.addTable([hdr, ...N.models.map((m, i) => {
    const o = { fill: { color: isOurs(m) ? "FBE3D9" : (i % 2 ? HEX.lt1 : HEX.lt2) }, bold: isOurs(m), color: HEX.dk1 };
    return [{ text: m.name, options: o }, ...[m.params_M.toFixed(1), m.latency_ms.toFixed(1), m.fps.toFixed(1), String(m.vram_MB), fmt3(m.mcorr)]
      .map((v) => ({ text: v, options: { ...o, align: "right" } }))];
  })], { x: M, y: 1.35, w: 8.3, colW: [2.0, 1.15, 1.3, 0.9, 1.45, 1.5], fontSize: 14, fontFace: THEME.bodyFontFace, rowH: 0.5, border: { type: "none" }, valign: "middle", margin: [0, 0.1, 0, 0.1], objectName: "efficiency table" });
  s.addText(`Batch size 1, image 768×1024, ${N.gpu}. SAM-based models and ours in fp16, CNNs in fp32.`,
    { x: M, y: 5.5, w: 8.3, h: 0.6, fontSize: 12, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "efficiency note" });
  stat(s, 9.3, 1.4, W - M - 9.3, FIVE.latency_ms.toFixed(1) + " ms", `per image for Ours (512): ${FIVE.fps.toFixed(0)} FPS`, "stat latency");
  stat(s, 9.3, 3.2, W - M - 9.3, FIVE.vram_MB + " MB", "peak GPU memory for Ours (512)", "stat vram", C.accent1);
  stat(s, 9.3, 5.0, W - M - 9.3, N.train_hours["Ours-Full-512"] + " h", `to train Ours (512) on the 4 GB GPU (1024: ${N.train_hours["Ours-Full"]} h)`, "stat train", C.text1);
  s.addNotes(`All timings were measured on our own RTX 3050 laptop GPU. Ours (512) is the fastest model in the table and uses the least memory, while beating SINet-V2 under corruption (+${S.five_sinet.diff.toFixed(3)}). SAM-Adapter is the most accurate on clean images but takes about ${(model("SAM-Adapter").latency_ms / 1000).toFixed(1)} s per image.`);

  // 13 ----------------------------------------------------------- performance: all datasets
  s = contentSlide("Performance on all four test sets", "Performance");
  const dsets = N.datasets;
  const best = Object.fromEntries(dsets.map((d) => [d, Math.max(...N.models.map((m) => m.mcorr_ds[d] ?? -1))]));
  const bestSpec = Object.fromEntries(dsets.map((d) => [d, Math.max(...["SINet-V2", "PFNet", "ZoomNet"].map((k) => model(k).mcorr_ds[d]))]));
  s.addTable([[head("Sα under corruption"), ...dsets.map((d) => ({ ...head(d), options: { ...head(d).options, align: "right" } }))],
    ...N.models.map((m, i) => {
      const o = { fill: { color: isOurs(m) ? "FBE3D9" : (i % 2 ? HEX.lt1 : HEX.lt2) }, color: HEX.dk1 };
      return [{ text: m.name, options: { ...o, bold: isOurs(m) } }, ...dsets.map((d) => {
        const v = m.mcorr_ds[d];
        return { text: fmt3(v), options: { ...o, align: "right", bold: v === best[d], color: isOurs(m) && v > bestSpec[d] ? HEX.accent2 : HEX.dk1 } };
      })];
    })], { x: M, y: 1.35, w: 8.3, colW: [2.7, 1.4, 1.4, 1.4, 1.4], fontSize: 14, fontFace: THEME.bodyFontFace, rowH: 0.5, border: { type: "none" }, valign: "middle", margin: [0, 0.1, 0, 0.1], objectName: "datasets table" });
  s.addText("Bold: best per dataset. Orange: our model scores higher than every specialist model. SAM-Adapter was not run on NC4K (≈4 s per image × 4,121 images × 41 conditions).",
    { x: M, y: 5.5, w: 8.3, h: 0.8, fontSize: 12, color: C.accent6, margin: 0, valign: "top", isTextBox: true, objectName: "datasets note" });
  card(s, 9.3, 1.4, W - M - 9.3, 4.9, "datasets takeaway card");
  s.addText([
    { text: "What the table says", options: { bold: true, fontSize: 18, color: HEX.dk1, breakLine: true } },
    { text: " ", options: { fontSize: 8, breakLine: true } },
    { text: "Ours (1024) is ahead of all specialists on COD10K and NC4K, the two large test sets.", options: { bullet: true, breakLine: true } },
    { text: "Level with SINet-V2 on CAMO, behind it on CHAMELEON (76 images).", options: { bullet: true, breakLine: true } },
    { text: "SAM2-UNet is best everywhere, at 20× our size.", options: { bullet: true } },
  ], { x: 9.55, y: 1.6, w: W - M - 9.8, h: 4.5, fontSize: 15, color: C.text2, margin: 0, valign: "top", paraSpaceAfter: 8, isTextBox: true, objectName: "datasets takeaway" });
  s.addNotes(`Mean Sα over the 40 corrupted conditions on each dataset. On NC4K, Ours (512) beats SINet-V2 by +${S.five_sinet_nc4k.diff.toFixed(3)} (${ci(S.five_sinet_nc4k)}). CHAMELEON has only 76 images, so its differences are less certain.`);

  // 14 ----------------------------------------------------------- contribution
  pres.addSection({ title: "Contribution" });
  s = contentSlide("Contribution: what is new compared with the literature", "Contribution");
  const contribs = [
    ["First corruption benchmark with SAM-based COD models",
      "The only earlier COD corruption benchmark (COD10K-C, 2026) tested CNN specialists only. We add SAM2-UNet and SAM-Adapter: 7 models, 4 datasets, 41 conditions each."],
    ["Held-out (unseen) corruption protocol",
      `Train on 4 corruptions, test on 4 never seen, including a new rain corruption. Gain on unseen: ${signed(S.full_clean_unseen.diff)} Sα (${ci(S.full_clean_unseen)}); rain ${signed(S.full_clean_rain.diff)}.`],
    ["Input resolution explains robustness",
      `The 1024-px models lose the most under local corruptions. Training at 512 px keeps accuracy there (${signed(S.five_ours_local.diff)}, ${ci(S.five_ours_local)}) at twice the speed.`],
    ["A robust adapter that trains on a 4 GB laptop",
      `Prompt-free MobileSAM adapter with a clean/corrupted consistency loss and ${mParams(N.params.trainable)} M trained parameters. Ours (1024) is ahead of all specialists under corruption on COD10K and NC4K; Ours (512) trains in ${N.train_hours["Ours-Full-512"]} h.`],
  ];
  const cw2 = (CW - 0.3) / 2, ch2 = 2.55;
  contribs.forEach(([h, b], i) => {
    const x = M + (i % 2) * (cw2 + 0.3), y = 1.4 + Math.floor(i / 2) * (ch2 + 0.3);
    card(s, x, y, cw2, ch2, `contribution ${i + 1}`);
    s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: `contribution number bg ${i + 1}` });
    s.addText(String(i + 1), { x: x + 0.25, y: y + 0.25, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `contribution number ${i + 1}` });
    s.addText(h, { x: x + 1.05, y: y + 0.2, w: cw2 - 1.3, h: 0.7, fontSize: 18, bold: true, color: C.text1, margin: 0, valign: "middle", isTextBox: true, objectName: `contribution head ${i + 1}` });
    s.addText(b, { x: x + 0.25, y: y + 1.0, w: cw2 - 0.5, h: ch2 - 1.15, fontSize: 16, color: C.text2, margin: 0, valign: "top", isTextBox: true, objectName: `contribution body ${i + 1}` });
  });
  s.addNotes("These four points are not in the literature. COD10K-C (Sayem, arXiv 2606.02603) is the closest work; it has no SAM-based models, no unseen-corruption split and no resolution analysis. Our model is not the most accurate overall, so the claim is robustness per unit of compute, not state of the art.");

  // 15 ----------------------------------------------------------- summary
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Contribution" });
  s.addText("Summary", { x: M, y: 0.6, w: CW, h: 0.9, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: "summary title" });
  const take = [
    ["SAM2-UNet is the most accurate model, also under corruption", `at ${model("SAM2-UNet").params_M} M parameters; SAM-Adapter takes ${(model("SAM-Adapter").latency_ms / 1000).toFixed(1)} s per image on a laptop GPU.`],
    ["Our 512-px adapter is the fastest model tested", `${FIVE.latency_ms.toFixed(1)} ms and ${FIVE.vram_MB} MB, and ahead of SINet-V2 under corruption (${signed(S.five_sinet.diff)}).`],
    ["Robustness partly transfers to unseen corruptions", `${signed(S.full_clean_unseen.diff)} Sα from training on four corruption types.`],
  ];
  take.forEach(([h, b], i) => {
    const y = 1.9 + i * 1.35;
    s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: `summary dot ${i + 1}` });
    s.addText(String(i + 1), { x: M, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `summary number ${i + 1}` });
    s.addText([{ text: h, options: { bold: true, fontSize: 20, color: HEX.lt1, breakLine: true } }, { text: b, options: { fontSize: 16, color: HEX.lt2 } }],
      { x: M + 0.8, y, w: CW - 0.8, h: 1.1, margin: 0, valign: "top", isTextBox: true, objectName: `summary item ${i + 1}` });
  });
  s.addText("Limitations: synthetic corruptions are not real camera degradation; SAM2-UNet remains more accurate; one GPU class tested.",
    { x: M, y: 6.1, w: CW, h: 0.6, fontSize: 14, italic: true, color: C.accent3, margin: 0, isTextBox: true, objectName: "limitations" });
  s.addNotes("Close with the three takeaways and the limitations. Offer to run the live demo: python demo/app.py.");

  await pres.writeFile({ fileName: OUT });
  if (APPLY_THEME) await require(APPLY_THEME).applyTheme(OUT, THEME);
  console.log("wrote " + path.relative(REPO, OUT));
}

build().catch((e) => { console.error(e); process.exit(1); });
