"""Generate the two-page report from measured artifacts (requires reportlab)."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                               PageBreak, Image, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))

scripted = read("results/scripted_demo.json")[0]
random = read("results/random_demo.json")[0]
training = read("results/sac_smoke/training_summary.json")
manifest = read("results/environment_manifest.json")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", fontName="Helvetica", fontSize=10.1, leading=13.2,
                          spaceAfter=7, textColor=colors.black))
styles.add(ParagraphStyle(name="SmallText", parent=styles["Body"], fontSize=8.2, leading=10.4, spaceAfter=4))
styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=11.4, leading=14,
                          spaceBefore=7, spaceAfter=5, textColor=colors.black))
styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=17, leading=20,
                          spaceAfter=5, textColor=colors.black))
styles.add(ParagraphStyle(name="Caption", parent=styles["SmallText"], alignment=TA_CENTER))

story = []
def para(text, style="Body"):
    story.append(Paragraph(text, styles[style]))
def section(title, text):
    story.append(KeepTogether([Paragraph(title, styles["Section"]), Paragraph(text, styles["Body"])]))

para("Adaptive Reward Shaping for Robotic Grasping", "ReportTitle")
para("ME5418 - Group 36 | Environment Implementation | 15 October 2026", "SmallText")
section("1. Scope and implementation", "We implement a Gymnasium environment for a Franka Panda arm to reach, grasp and lift a cube. MuJoCo supplies dynamics; robosuite supplies the robot, OSC controller and Lift scene. Our additions are the Gymnasium adapter, three reward components, episode-stage tracking, an adaptive scheduler, replay reward relabelling, tests and demos. This milestone validates the environment; it does not establish that adaptive rewards improve learning.")

figures = Table([[Image(str(ROOT / "results/panda_overview.png"), width=194, height=145.5),
                  Image(str(ROOT / "results/scripted_final.png"), width=194, height=145.5)]], colWidths=[234,234])
figures.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                            ("TOPPADDING",(0,0),(-1,-1),1),("BOTTOMPADDING",(0,0),(-1,-1),1)]))
story.append(figures)
para("Figure 1. Simulated Panda and cube (left); successful scripted lift (right). Images are simulator renders, not evidence of an RL-trained policy.", "Caption")

section("2. Observation, action and episode contract", "The observation is a 60-element float32 vector: 50 robot proprioceptive values and 10 object values. It includes joint, end-effector and gripper states, cube pose and relative position; no camera image enters the policy. Actions are seven values in [-1, 1]: world-frame position and rotation increments plus gripper command. OSC_POSE scales position to at most 0.05 m and rotation to 0.5 rad per command; control runs at 20 Hz with a 0.002 s physics step.")
para("reset(seed) returns (observation, info); step(action) returns (observation, reward, terminated, truncated, info). Success terminates; reaching the 200-step horizon without success truncates. Invalid or nonfinite actions are rejected and finite out-of-range actions are clipped. Model construction uses seed 0 to keep cube dimensions identical; episode seeds randomize placement. Resetting does not erase the curriculum unless explicitly requested.")
section("3. Reward and stage definitions", "Reach reward is 1 - tanh(5d), where d is end-effector-to-cube distance in metres; reaching within 0.05 m records stage success. Grasp reward is 1 only when robosuite detects cube contact with both finger contact groups. Gripper closure alone is insufficient. Lift progress is clip((z - z_reset)/0.04, 0, 1), overridden to 1 on task success. The authoritative Lift test is robosuite's cube-centre height greater than table height + 0.04 m (here z &gt; 0.84 m), not a 4 cm rise from the reset position.")
para("Sparse mode gives only task-success reward. Fixed mode uses weights (0.3, 0.3, 0.4). Adaptive mode uses the same components with scheduler-selected weights. Constants are explicit initial design choices; they have not been optimized by a large pilot study.")

story.append(PageBreak())
para("Adaptive scheduling, verification and reflection", "ReportTitle")
section("4. Scheduler and replay handling", "A full window of the latest N=20 completed episodes estimates the fraction that ever reached, grasped or lifted. Every K=5 completed episodes, reaching at least 80% selects grasp emphasis; reaching at least 80% and grasping at least 60% selects lift emphasis. Candidate stages require two consecutive confirmations before a switch, including regression after sustained failures. Incomplete episodes are excluded. Lift success rate is logged, rather than used as another switching gate.")
para("Stage weights are (0.6, 0.3, 0.1), (0.2, 0.6, 0.2) and (0.1, 0.2, 0.7). Each episode keeps its starting weights; updates take effect at the next reset. For adaptive SAC, ComponentReplayBuffer stores all three reward components and recomputes scalar rewards using current weights when sampled. Timeout transitions retain bootstrapping semantics. This implementation supports single-environment, one-step SAC without VecNormalize; historical scalar rewards remain an explicit optional ablation.")
section("5. Executed validation", "Both Gymnasium and Stable-Baselines3 environment checkers passed. Nine automated tests cover reward ordering, all three modes, seeded resets, identical geometry across instances, truncation, empty-gripper contact rejection, simulator success termination, real scripted contact/lift, curriculum transitions and replay relabelling. Infinite observation bounds produce two checker warnings; returned observations are finite. Native human rendering and offscreen MP4 recording also ran successfully.")

rows = [["Executed check", "Measured result"],
        ["Random policy, seed 42", f"{random['steps']} steps; timeout; no task success"],
        ["Scripted policy, seed 42", f"{scripted['steps']} steps; Reach / Grasp / Lift achieved"],
        ["Adaptive SAC smoke run", f"{training['steps']} interactions; {training['gradient_updates']} gradient updates"],
        ["Recorded demo process RAM", f"Random {random['peak_process_ram_mib']:.0f} MiB; scripted {scripted['peak_process_ram_mib']:.0f} MiB RSS"]]
table = Table([[Paragraph(c, styles["SmallText"]) for c in row] for row in rows], colWidths=[185,283])
table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dce7ef")),
    ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#d9d9d9")),
    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("LEFTPADDING",(0,0),(-1,-1),8),
    ("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),5),
    ("BOTTOMPADDING",(0,0),(-1,-1),5)]))
story.append(table)
story.append(Spacer(1,7))
para("The saved smoke model was reloaded and evaluated on one independent seed: no successful lift. This is a pipeline check, not a success-rate estimate. Results, videos, traces and test logs are included. A separate fresh Conda environment was rebuilt from environment.yml and the full pip lock; its validation logs accompany the submission.")
section("6. Limitations and next experiments", "Bilateral contact is a grasp proxy, not proof of force closure or long-term stability. The scripted servo reads simulator state and is not learned. The adaptive reward is nonstationary and its weights are not policy observations; relabelling alone does not prove stable learning. We have not run a controlled multi-seed comparison, vision, or sim-to-real experiments. Future work should freeze pilot parameters, train all three conditions with identical budgets and at least five seeds where feasible, and compare common Lift success rate and sample efficiency, rather than raw returns with different scales.")
para("Reproducibility: Python 3.10.22; robosuite 1.5.2; MuJoCo 3.3.7; Gymnasium 1.2.1; SB3 2.7.0; NumPy 1.26.4. Windows x64 was tested. A newer MuJoCo release broke the controller mass-matrix interface, motivating the pinned version. Other operating systems remain unverified.", "SmallText")
para("References: robosuite documentation, robosuite.ai/docs/; Gymnasium Env API, gymnasium.farama.org/api/env/; SB3 SAC documentation, stable-baselines3.readthedocs.io/en/v2.7.0/modules/sac.html. Exact installed Lift success implementation is identified in README.", "SmallText")
para("AI use: Codex/ChatGPT assisted code, tests and writing; execution evidence is included. Group members remain responsible for understanding, checking and disclosing the submitted work according to course rules.", "SmallText")

def footer(canvas, doc):
    canvas.setFont("Helvetica",8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(54,30,"ME5418 / Group 36 - Environment milestone")
    canvas.drawRightString(558,30,f"{doc.page}")

doc = SimpleDocTemplate(str(ROOT / "report.pdf"), pagesize=(612,792),
    rightMargin=54, leftMargin=54, topMargin=40, bottomMargin=46,
    title="Adaptive Reward Shaping for Robotic Grasping", author="Group 36")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
from pypdf import PdfReader
pages = len(PdfReader(ROOT / "report.pdf").pages)
assert pages == 2, f"Expected exactly two pages; got {pages}"
print(f"Created report.pdf: {pages} pages")
