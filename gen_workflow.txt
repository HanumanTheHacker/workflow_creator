#!/usr/bin/env python3
"""
gen_workflow.py
---------------
Generate a ComfyUI LTX2 workflow JSON with N keyframes.

Usage:
    python gen_workflow.py --keyframes 4 --times 0 3 7 11 --input ltx2_base.json --output my_workflow.json

Arguments:
    --keyframes   Number of keyframes (must match length of --times)
    --times       Start time in seconds for each keyframe (space-separated)
                  First value is always frame 0 (ignored in calc, always 0)
    --input       Path to the base workflow JSON  (default: ltx2_3_1_1_influencer_native_audio_V7.json)
    --output      Output JSON filename            (default: workflow_Nkf.json)

Examples:
    python gen_workflow.py --keyframes 3 --times 0 5 10
    python gen_workflow.py --keyframes 5 --times 0 2 5 8 12 --output my5frame.json
"""

import json
import copy
import argparse
import sys
import os

# ─────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────
ROW_HEIGHT      = 360   # vertical gap between keyframe rows
LOAD_X          = -1600
RESIZE_X        = -1280
GETW_X          = -1280
GETH_X          = -1280
PREP_X          = -940
GETVAE_X        = -940
GETFPS_X        = -1280
PRIMFLOAT_X     = -1280
CALC_X          = -940
GUIDE_X         = -620

LOAD_Y_START    = 5500   # Y position of first non-zero keyframe row (row index 1)
LOAD_W, LOAD_H  = 280, 320
RESIZE_W        = 290
PREP_W          = 260
GETVAE_W        = 210
GETFPS_W        = 210
PRIMFLOAT_W     = 260
CALC_W          = 263
GUIDE_W, GUIDE_H = 280, 174

# Upstream node IDs (fixed, from base workflow)
NODE_CONDITIONING_POS_SRC = 107   # LTXVConditioning out_slot 0 → positive
NODE_CONDITIONING_NEG_SRC = 107   # LTXVConditioning out_slot 1 → negative
NODE_LATENT_SRC           = 108   # EmptyLTXVLatentVideo out_slot 0 → latent
NODE_FIRST_IMAGE_SRC      = 162   # LTXVPreprocess (first frame preprocess) out_slot 0
NODE_VAE_SRC              = 218   # GetNode Get_vae out_slot 0

# Downstream node IDs (fixed)
NODE_SET_POSITIVE = 226   # SetNode Set_positive  in_slot 0
NODE_SET_NEGATIVE = 227   # SetNode Set_negative  in_slot 0
NODE_CONCAT       = 109   # LTXVConcatAVLatent    in_slot 0


def load_base(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def next_id(state: dict) -> int:
    state["last_node_id"] += 1
    return state["last_node_id"]


def next_link(state: dict) -> int:
    state["last_link_id"] += 1
    return state["last_link_id"]


def find_node(nodes: list, node_id: int) -> dict:
    for n in nodes:
        if n["id"] == node_id:
            return n
    raise ValueError(f"Node {node_id} not found")


def remove_link_from_node_input(nodes: list, node_id: int, slot: int):
    """Clear a link reference from a node's input slot."""
    n = find_node(nodes, node_id)
    if "inputs" in n and slot < len(n["inputs"]):
        n["inputs"][slot]["link"] = None


def remove_link_from_node_output(nodes: list, node_id: int, slot: int, link_id: int):
    """Remove a specific link_id from a node's output slot links list."""
    n = find_node(nodes, node_id)
    if "outputs" in n and slot < len(n["outputs"]):
        lks = n["outputs"][slot].get("links") or []
        if link_id in lks:
            lks.remove(link_id)


def add_output_link(nodes: list, node_id: int, slot: int, link_id: int):
    n = find_node(nodes, node_id)
    lks = n["outputs"][slot].setdefault("links", []) or []
    if lks is None:
        n["outputs"][slot]["links"] = [link_id]
    else:
        lks.append(link_id)


def set_input_link(nodes: list, node_id: int, slot: int, link_id: int):
    n = find_node(nodes, node_id)
    n["inputs"][slot]["link"] = link_id


def make_node(nid, ntype, title, pos, size, widgets_values, inputs, outputs):
    return {
        "id": nid,
        "type": ntype,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": 0,
        "mode": 0,
        "title": title,
        "widgets_values": widgets_values,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": ntype},
    }


def build_link(lid, from_node, from_slot, to_node, to_slot, ltype):
    return [lid, from_node, from_slot, to_node, to_slot, ltype]


def strip_old_keyframe_nodes(data: dict) -> dict:
    """
    Remove all dynamically-generated keyframe nodes (mid/last LoadImage,
    Resize, Preprocess, GetVAE, GetFPS, PrimitiveFloat, Calculator, AddGuide)
    and their links, leaving the first-frame setup and downstream nodes intact.
    Also disconnects the upstream/downstream dangling links.
    """
    REMOVABLE_IDS = {
        621, 622,           # LoadImage mid/last
        623, 624,           # ResizeImageMaskNode mid/last
        629, 630,           # LTXVPreprocess mid/last
        631, 632,           # GetNode Get_vae (mid/last)
        625, 626, 627, 628, # GetNode Get_width/height (mid/last)
        637, 641,           # GetNode Get_fps (mid/last)
        636, 640,           # PrimitiveFloat time (mid/last)
        638, 642,           # SimpleCalculatorKJ (mid/last)
        634, 635,           # LTXVAddGuide mid/last
    }

    # Collect link IDs that involve removable nodes
    bad_links = set()
    for lnk in data["links"]:
        if lnk[1] in REMOVABLE_IDS or lnk[3] in REMOVABLE_IDS:
            bad_links.add(lnk[0])

    data["links"] = [lnk for lnk in data["links"] if lnk[0] not in bad_links]
    data["nodes"] = [n for n in data["nodes"] if n["id"] not in REMOVABLE_IDS]

    # Fix node 633 (first AddGuide): clear outputs that pointed to old mid node
    n633 = find_node(data["nodes"], 633)
    n633["outputs"][0]["links"] = []   # positive
    n633["outputs"][1]["links"] = []   # negative
    n633["outputs"][2]["links"] = []   # latent

    return data


def add_keyframe_nodes(data: dict, times_sec: list) -> dict:
    """
    Add keyframe node groups for frames 1..N-1 (frame 0 is already wired).
    times_sec[0] is ignored (always frame_idx=0, handled by node 633).
    """
    nodes  = data["nodes"]
    links  = data["links"]
    state  = data  # mutated in-place for last_node_id / last_link_id

    num_frames = len(times_sec)

    # Track the AddGuide chain: starts at node 633 (first frame)
    prev_guide_id = 633

    for i in range(1, num_frames):
        time_sec = times_sec[i]
        row      = i - 1                        # row 0 = first non-zero keyframe
        y_base   = LOAD_Y_START + row * ROW_HEIGHT
        label    = ordinal(i + 1)               # "2nd", "3rd", …

        # ── Allocate node IDs ──────────────────────────────────────
        nid_load      = next_id(state)
        nid_resize    = next_id(state)
        nid_getw      = next_id(state)
        nid_geth      = next_id(state)
        nid_prep      = next_id(state)
        nid_getvae    = next_id(state)
        nid_getfps    = next_id(state)
        nid_primfloat = next_id(state)
        nid_calc      = next_id(state)
        nid_guide     = next_id(state)

        # ── Allocate link IDs ──────────────────────────────────────
        lid_load_resize   = next_link(state)   # LoadImage → Resize input
        lid_getw_resize   = next_link(state)   # GetWidth  → Resize width
        lid_geth_resize   = next_link(state)   # GetHeight → Resize height
        lid_resize_prep   = next_link(state)   # Resize → Preprocess
        lid_prep_guide    = next_link(state)   # Preprocess → Guide image
        lid_getvae_guide  = next_link(state)   # GetVAE → Guide vae
        lid_primfloat_calc= next_link(state)   # PrimitiveFloat → Calc a
        lid_getfps_calc   = next_link(state)   # GetFPS → Calc b
        lid_calc_guide    = next_link(state)   # Calc INT → Guide frame_idx
        # Chain from previous guide
        lid_prev_pos      = next_link(state)   # prev_guide pos → this guide pos
        lid_prev_neg      = next_link(state)   # prev_guide neg → this guide neg
        lid_prev_lat      = next_link(state)   # prev_guide lat → this guide lat

        # ── Build nodes ────────────────────────────────────────────

        node_load = make_node(
            nid_load, "LoadImage", f"Load Frame {i+1}",
            pos=[LOAD_X, y_base], size=[LOAD_W, LOAD_H],
            widgets_values=[f"frame{i+1}.jpeg", "image"],
            inputs=[],
            outputs=[
                {"name": "IMAGE", "type": "IMAGE", "links": [lid_load_resize]},
                {"name": "MASK",  "type": "MASK",  "links": None},
            ],
        )

        node_getw = make_node(
            nid_getw, "GetNode", "Get_width",
            pos=[GETW_X, y_base], size=[GETVAE_W, 34],
            widgets_values=["width"], inputs=[],
            outputs=[{"name": "INT", "type": "INT", "links": [lid_getw_resize]}],
        )

        node_geth = make_node(
            nid_geth, "GetNode", "Get_height",
            pos=[GETH_X, y_base + 50], size=[GETVAE_W, 34],
            widgets_values=["height"], inputs=[],
            outputs=[{"name": "INT", "type": "INT", "links": [lid_geth_resize]}],
        )

        node_resize = make_node(
            nid_resize, "ResizeImageMaskNode", f"Resize Frame {i+1}",
            pos=[RESIZE_X, y_base + 100], size=[RESIZE_W, 154],
            widgets_values=["scale dimensions", 640, 640, "center", "nearest-exact"],
            inputs=[
                {"name": "input",             "type": "IMAGE", "link": lid_load_resize},
                {"name": "resize_type.width", "type": "INT",   "link": lid_getw_resize},
                {"name": "resize_type.height","type": "INT",   "link": lid_geth_resize},
            ],
            outputs=[{"name": "resized", "type": "IMAGE", "links": [lid_resize_prep]}],
        )

        node_prep = make_node(
            nid_prep, "LTXVPreprocess", f"Preprocess Frame {i+1}",
            pos=[PREP_X, y_base + 100], size=[PREP_W, 66],
            widgets_values=[33],
            inputs=[{"name": "image", "type": "IMAGE", "link": lid_resize_prep}],
            outputs=[{"name": "output_image", "type": "IMAGE", "links": [lid_prep_guide]}],
        )

        node_getvae = make_node(
            nid_getvae, "GetNode", "Get_vae",
            pos=[GETVAE_X, y_base + 200], size=[GETVAE_W, 34],
            widgets_values=["vae"], inputs=[],
            outputs=[{"name": "VAE", "type": "VAE", "links": [lid_getvae_guide]}],
        )

        node_primfloat = make_node(
            nid_primfloat, "PrimitiveFloat", f"Frame {i+1} Time (sec)",
            pos=[PRIMFLOAT_X, y_base + 260], size=[PRIMFLOAT_W, 66],
            widgets_values=[float(time_sec)],
            inputs=[],
            outputs=[{"name": "FLOAT", "type": "FLOAT", "links": [lid_primfloat_calc]}],
        )

        node_getfps = make_node(
            nid_getfps, "GetNode", "Get_fps",
            pos=[GETFPS_X, y_base + 310], size=[GETFPS_W, 34],
            widgets_values=["fps"], inputs=[],
            outputs=[{"name": "FLOAT", "type": "FLOAT", "links": [lid_getfps_calc]}],
        )

        node_calc = make_node(
            nid_calc, "SimpleCalculatorKJ",
            f"frame{i+1}_idx = round(sec × fps)",
            pos=[CALC_X, y_base + 260], size=[CALC_W, 188],
            widgets_values=["round(a*b)"],
            inputs=[
                {"name": "variables.a", "type": "FLOAT", "link": lid_primfloat_calc},
                {"name": "variables.b", "type": "FLOAT", "link": lid_getfps_calc},
                {"name": "variables.c", "type": "FLOAT", "link": None},
                {"name": "a",           "type": "FLOAT", "link": None},
                {"name": "b",           "type": "FLOAT", "link": None},
            ],
            outputs=[
                {"name": "FLOAT",   "type": "FLOAT",   "links": []},
                {"name": "INT",     "type": "INT",     "links": [lid_calc_guide]},
                {"name": "BOOLEAN", "type": "BOOLEAN", "links": None},
            ],
        )

        # AddGuide inputs: pos(0), neg(1), vae(2), latent(3), image(4), frame_idx(5)
        guide_inputs = [
            {"name": "positive",  "type": "CONDITIONING", "link": lid_prev_pos},
            {"name": "negative",  "type": "CONDITIONING", "link": lid_prev_neg},
            {"name": "vae",       "type": "VAE",           "link": lid_getvae_guide},
            {"name": "latent",    "type": "LATENT",        "link": lid_prev_lat},
            {"name": "image",     "type": "IMAGE",         "link": lid_prep_guide},
            {"name": "frame_idx", "type": "INT",
             "widget": {"name": "frame_idx"}, "link": lid_calc_guide},
        ]

        node_guide = make_node(
            nid_guide, "LTXVAddGuide", f"LTXVAddGuide (Frame {i+1})",
            pos=[GUIDE_X, y_base], size=[GUIDE_W, GUIDE_H],
            widgets_values=[0.7, 1],
            inputs=guide_inputs,
            outputs=[
                {"name": "positive", "type": "CONDITIONING", "links": []},
                {"name": "negative", "type": "CONDITIONING", "links": []},
                {"name": "latent",   "type": "LATENT",       "links": []},
            ],
        )

        # ── Wire chain from previous guide ────────────────────────
        prev_guide_node = find_node(nodes, prev_guide_id)
        prev_guide_node["outputs"][0]["links"].append(lid_prev_pos)
        prev_guide_node["outputs"][1]["links"].append(lid_prev_neg)
        prev_guide_node["outputs"][2]["links"].append(lid_prev_lat)

        # ── Add link records ───────────────────────────────────────
        links.extend([
            build_link(lid_load_resize,    nid_load,      0, nid_resize,    0, "IMAGE"),
            build_link(lid_getw_resize,    nid_getw,      0, nid_resize,    1, "INT"),
            build_link(lid_geth_resize,    nid_geth,      0, nid_resize,    2, "INT"),
            build_link(lid_resize_prep,    nid_resize,    0, nid_prep,      0, "IMAGE"),
            build_link(lid_prep_guide,     nid_prep,      0, nid_guide,     4, "IMAGE"),
            build_link(lid_getvae_guide,   nid_getvae,    0, nid_guide,     2, "VAE"),
            build_link(lid_primfloat_calc, nid_primfloat, 0, nid_calc,      0, "FLOAT"),
            build_link(lid_getfps_calc,    nid_getfps,    0, nid_calc,      1, "FLOAT"),
            build_link(lid_calc_guide,     nid_calc,      1, nid_guide,     5, "INT"),
            build_link(lid_prev_pos,       prev_guide_id, 0, nid_guide,     0, "CONDITIONING"),
            build_link(lid_prev_neg,       prev_guide_id, 1, nid_guide,     1, "CONDITIONING"),
            build_link(lid_prev_lat,       prev_guide_id, 2, nid_guide,     3, "LATENT"),
        ])

        # ── Add nodes ──────────────────────────────────────────────
        nodes.extend([
            node_load, node_getw, node_geth, node_resize,
            node_prep, node_getvae, node_getfps, node_primfloat,
            node_calc, node_guide,
        ])

        prev_guide_id = nid_guide

    # ── Wire last guide → downstream nodes (Set_positive, Set_negative, Concat)
    last_guide = find_node(nodes, prev_guide_id)

    lid_final_pos = next_link(state)
    lid_final_neg = next_link(state)
    lid_final_lat = next_link(state)

    last_guide["outputs"][0]["links"].append(lid_final_pos)
    last_guide["outputs"][1]["links"].append(lid_final_neg)
    last_guide["outputs"][2]["links"].append(lid_final_lat)

    links.extend([
        build_link(lid_final_pos, prev_guide_id, 0, NODE_SET_POSITIVE, 0, "CONDITIONING"),
        build_link(lid_final_neg, prev_guide_id, 1, NODE_SET_NEGATIVE, 0, "CONDITIONING"),
        build_link(lid_final_lat, prev_guide_id, 2, NODE_CONCAT,       0, "LATENT"),
    ])

    # Fix downstream node inputs
    set_input_link(nodes, NODE_SET_POSITIVE, 0, lid_final_pos)
    set_input_link(nodes, NODE_SET_NEGATIVE, 0, lid_final_neg)
    set_input_link(nodes, NODE_CONCAT,       0, lid_final_lat)

    return data


def ordinal(n: int) -> str:
    suffixes = {1: "st", 2: "nd", 3: "rd"}
    return f"{n}{suffixes.get(n if n < 20 else n % 10, 'th')}"


def update_markdown_note(data: dict, num_frames: int, times_sec: list):
    """Update the 3-Frame Keyframe Guide markdown note to reflect actual count."""
    for n in data["nodes"]:
        if n["type"] == "MarkdownNote" and "Keyframe" in n.get("title", ""):
            lines = [f"## 🎬 {num_frames}-Frame Keyframe Conditioning\n"]
            lines.append("Each image opens its section — the sampler interpolates between them.\n")
            for i in range(num_frames):
                sec  = times_sec[i]
                label = "Start" if i == 0 else ("End" if i == num_frames - 1 else f"Mid {i}")
                lines.append(f"**Frame {i+1} ({label})** — `time = {sec}s`  ")
                if i == 0:
                    lines.append("→ Uses existing first-frame **LoadImage** at top of workflow.\n")
                else:
                    lines.append(f"→ Upload image in **Load Frame {i+1}**. Set **Frame {i+1} Time (sec)** = {sec}.\n")
            n["widgets_values"] = ["".join(lines)]
            n["title"] = f"{num_frames}-Frame Keyframe Guide (LTXVAddGuide)"
            break


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a ComfyUI LTX2 workflow JSON with N keyframes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--keyframes", type=int, required=True,
                        help="Number of keyframes (must match length of --times)")
    parser.add_argument("--times", type=float, nargs="+", required=True,
                        help="Start time in seconds for each keyframe (space-separated). "
                             "First value is always 0 (first frame).")
    parser.add_argument("--input", default="ltx2_3_1_1_influencer_native_audio_V7.json",
                        help="Path to base workflow JSON")
    parser.add_argument("--output", default=None,
                        help="Output filename (default: workflow_Nkf.json)")
    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────
    if args.keyframes < 1:
        sys.exit("Error: --keyframes must be at least 1")
    if len(args.times) != args.keyframes:
        sys.exit(f"Error: --keyframes {args.keyframes} but {len(args.times)} times provided. "
                 "They must match.")
    if args.times[0] != 0:
        print(f"Warning: First keyframe time is {args.times[0]}s, "
              "but the first frame is always frame_idx=0. Ignoring first time value.")
        args.times[0] = 0.0

    for i in range(1, len(args.times)):
        if args.times[i] <= args.times[i - 1]:
            sys.exit(f"Error: times must be strictly increasing. "
                     f"times[{i}]={args.times[i]} <= times[{i-1}]={args.times[i-1]}")

    if not os.path.exists(args.input):
        sys.exit(f"Error: Input file not found: {args.input}")

    output_path = args.output or f"workflow_{args.keyframes}kf.json"

    # ── Process ───────────────────────────────────────────────────
    print(f"Loading base workflow: {args.input}")
    data = load_base(args.input)

    print("Stripping existing keyframe nodes (mid/last)...")
    data = strip_old_keyframe_nodes(data)

    print(f"Building {args.keyframes}-keyframe chain with times: {args.times}")
    data = add_keyframe_nodes(data, args.times)

    update_markdown_note(data, args.keyframes, args.times)

    # ── Save ──────────────────────────────────────────────────────
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅  Done! Saved to: {output_path}")
    print(f"\nWorkflow summary:")
    print(f"  Keyframes : {args.keyframes}")
    for i, t in enumerate(args.times):
        label = "(first frame, always idx 0)" if i == 0 else f"→ frame_idx = round({t} × FPS)"
        print(f"  Frame {i+1}   : {t}s  {label}")
    print(f"\nIn ComfyUI:")
    print(f"  • Upload images to: Load Frame 1, Load Frame 2, ... Load Frame {args.keyframes}")
    if args.keyframes > 1:
        print(f"  • Time inputs visible as: 'Frame 2 Time (sec)', 'Frame 3 Time (sec)', ...")
    print()


if __name__ == "__main__":
    main()
