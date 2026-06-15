"""rosclaw-auto CLI."""
import argparse
import sys

from .config import AutoConfig
from .dashboard import DashboardExporter
from .engine import AutoEngine
from .reports import ReportGenerator


def _find_task_id(engine, task_name: str) -> str | None:
    """Find task by name or id."""
    # Try exact id match first
    t = engine.get_task(task_name)
    if t:
        return t.id
    # Try name match
    for task in engine.list_tasks():
        if task.name == task_name:
            return task.id
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw-auto", description="ROSClaw Self-Evolution Control Plane")
    sub = parser.add_subparsers(dest="cmd")

    # init
    p_init = sub.add_parser("init", help="Initialize an auto task")
    p_init.add_argument("--task", required=True)
    p_init.add_argument("--robot", default="panda")
    p_init.add_argument("--skill", required=True)
    p_init.add_argument("--env", default="maniskill")
    p_init.add_argument("--type", default="skill_tuning", choices=["skill_tuning", "failure_repair"])

    # run
    p_run = sub.add_parser("run", help="Run auto evolution")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--rounds", type=int, default=10)
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--policy", default="failure_guided")

    # status
    p_status = sub.add_parser("status", help="Show auto status")
    p_status.add_argument("--task", default=None)

    # champion
    p_champ = sub.add_parser("champion", help="Show current champion")
    p_champ.add_argument("--task", required=True)

    # deadends
    p_de = sub.add_parser("deadends", help="List dead ends")
    p_de.add_argument("--task", default=None)

    # report
    p_rep = sub.add_parser("report", help="Generate evolution report")
    p_rep.add_argument("--task", required=True)
    p_rep.add_argument("--output", default="")
    p_rep.add_argument("--format", default="md", choices=["md", "json"])

    # repair
    p_rep2 = sub.add_parser("repair", help="Repair from a failure")
    p_rep2.add_argument("--failure", required=True)

    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 1

    config = AutoConfig()
    engine = AutoEngine(config)

    if args.cmd == "init":
        task = engine.create_task(args.task, args.robot, args.skill, args.type, args.env)
        print(f"Created task: {task.id} | {task.name} | skill={task.target_skill_id}")
        return 0

    if args.cmd == "run":
        task_id = _find_task_id(engine, args.task)
        if not task_id:
            print(f"Task {args.task} not found. Run `rosclaw-auto init --task {args.task} ...` first.")
            return 1
        report = engine.run(task_id, args.rounds, args.dry_run)
        print(f"Evolution complete: {report.id}")
        print(f"  Proposals: {report.proposals_created}")
        print(f"  Experiments: {report.experiments_run}")
        print(f"  Champions: {report.champions_promoted}")
        print(f"  DeadEnds: {report.deadends_registered}")
        return 0

    if args.cmd == "status":
        tasks = engine.list_tasks()
        print(f"Auto tasks: {len(tasks)}")
        for t in tasks:
            print(f"  {t.id} | {t.name} | {t.status} | skill={t.target_skill_id}")
        return 0

    if args.cmd == "champion":
        task_id = _find_task_id(engine, args.task)
        champ = engine.get_champion(task_id or args.task)
        if champ:
            print(f"Champion: {champ.skill_id}")
            print(f"  Level: {champ.level}")
            print(f"  Metrics: {champ.metrics}")
        else:
            print("No champion found")
        return 0

    if args.cmd == "deadends":
        des = engine.list_deadends(args.task)
        print(f"Dead ends: {len(des)}")
        for d in des:
            print(f"  {d.id} | {d.direction} | {d.rejection_reason}")
        return 0

    if args.cmd == "report":
        task_id = _find_task_id(engine, args.task)
        task_id = task_id or args.task
        if getattr(args, "format", "md") == "json":
            exporter = DashboardExporter(engine)
            data = exporter.export_json(task_id)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(data)
                print(f"Dashboard JSON exported to {args.output}")
            else:
                print(data)
        else:
            gen = ReportGenerator(engine)
            md = gen.generate_markdown(task_id)
            if args.output:
                with open(args.output, "w") as f:
                    f.write(md)
                print(f"Markdown report written to {args.output}")
            else:
                print(md)
        return 0

    if args.cmd == "repair":
        print(f"Repair from failure {args.failure} (placeholder)")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
