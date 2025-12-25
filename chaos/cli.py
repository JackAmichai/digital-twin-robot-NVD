"""
Chaos Engineering CLI Runner.

Command-line interface for running chaos experiments.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Optional

from .engine import ChaosEngine, ExperimentState
from .experiments import (
    ALL_EXPERIMENTS,
    get_experiment_by_name,
    get_experiments_by_tag,
    list_experiments
)


def setup_logging(verbose: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def print_experiment_list(experiments, detailed: bool = False):
    """Print list of experiments."""
    print("\n" + "=" * 60)
    print("Available Chaos Experiments")
    print("=" * 60 + "\n")
    
    for exp in experiments:
        print(f"  📋 {exp.name}")
        if detailed:
            print(f"     Description: {exp.description}")
            print(f"     Tags: {', '.join(exp.tags)}")
            print(f"     Actions: {len(exp.actions)}")
            print()


def print_experiment_result(result):
    """Print experiment result in formatted output."""
    print("\n" + "=" * 60)
    print("Experiment Result")
    print("=" * 60)
    
    status_emoji = {
        ExperimentState.COMPLETED: "✅",
        ExperimentState.FAILED: "❌",
        ExperimentState.ROLLED_BACK: "⚠️",
        ExperimentState.ABORTED: "🛑"
    }
    
    emoji = status_emoji.get(result.final_state, "❓")
    
    print(f"\n  {emoji} Experiment: {result.experiment_name}")
    print(f"  Status: {result.final_state.value}")
    print(f"  Duration: {result.duration:.2f}s")
    print(f"  Start: {result.start_time.isoformat()}")
    print(f"  End: {result.end_time.isoformat()}")
    
    print("\n  Steady State Verification:")
    print(f"    Before: {'✓ PASS' if result.steady_state_before else '✗ FAIL'}")
    print(f"    After:  {'✓ PASS' if result.steady_state_after else '✗ FAIL'}")
    
    print("\n  Action Results:")
    for i, action_result in enumerate(result.action_results, 1):
        status = "✓" if action_result.get("success") else "✗"
        action_type = action_result.get("fault_type", "unknown")
        print(f"    {i}. [{status}] {action_type}")
        if action_result.get("error"):
            print(f"       Error: {action_result['error']}")
    
    if result.errors:
        print("\n  Errors:")
        for error in result.errors:
            print(f"    ⚠️  {error}")
    
    print("\n" + "=" * 60 + "\n")


async def run_experiment(engine: ChaosEngine, experiment_name: str, dry_run: bool = False):
    """Run a single experiment."""
    try:
        experiment = get_experiment_by_name(experiment_name)
    except ValueError as e:
        print(f"Error: {e}")
        return None
    
    print(f"\n🔬 Running experiment: {experiment.name}")
    print(f"   Description: {experiment.description}")
    
    if dry_run:
        print("\n   [DRY RUN] Would execute the following actions:")
        for i, action in enumerate(experiment.actions, 1):
            print(f"   {i}. {action.fault_type.value} on {action.target} for {action.duration}s")
        return None
    
    result = await engine.run_experiment(experiment)
    return result


async def run_experiments_by_tag(engine: ChaosEngine, tag: str, dry_run: bool = False):
    """Run all experiments with a specific tag."""
    experiments = get_experiments_by_tag(tag)
    
    if not experiments:
        print(f"No experiments found with tag: {tag}")
        return []
    
    print(f"\n🏷️  Running {len(experiments)} experiments with tag '{tag}'")
    
    results = []
    for experiment in experiments:
        result = await run_experiment(engine, experiment.name, dry_run)
        if result:
            results.append(result)
            print_experiment_result(result)
    
    return results


def export_results(results, output_file: str):
    """Export results to JSON file."""
    export_data = {
        "export_time": datetime.utcnow().isoformat(),
        "total_experiments": len(results),
        "results": []
    }
    
    for result in results:
        export_data["results"].append({
            "experiment_name": result.experiment_name,
            "final_state": result.final_state.value,
            "duration": result.duration,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "steady_state_before": result.steady_state_before,
            "steady_state_after": result.steady_state_after,
            "action_results": result.action_results,
            "errors": result.errors
        })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n📄 Results exported to: {output_file}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chaos Engineering CLI for Digital Twin Robotics Lab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available experiments
  python -m chaos.cli list

  # List experiments with details
  python -m chaos.cli list --detailed

  # Run a specific experiment
  python -m chaos.cli run voice-processing-latency

  # Dry run (show what would happen)
  python -m chaos.cli run voice-processing-latency --dry-run

  # Run all experiments with a tag
  python -m chaos.cli run-tag network

  # Export results to file
  python -m chaos.cli run voice-processing-latency --output results.json
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available experiments")
    list_parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed experiment information"
    )
    list_parser.add_argument(
        "--tag", "-t",
        type=str,
        help="Filter by tag"
    )
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "experiment",
        type=str,
        help="Name of the experiment to run"
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing"
    )
    run_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON)"
    )
    run_parser.add_argument(
        "--namespace", "-n",
        type=str,
        default="digital-twin",
        help="Kubernetes namespace (default: digital-twin)"
    )
    
    # Run-tag command
    tag_parser = subparsers.add_parser("run-tag", help="Run all experiments with a tag")
    tag_parser.add_argument(
        "tag",
        type=str,
        help="Tag to filter experiments"
    )
    tag_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing"
    )
    tag_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON)"
    )
    tag_parser.add_argument(
        "--namespace", "-n",
        type=str,
        default="digital-twin",
        help="Kubernetes namespace (default: digital-twin)"
    )
    
    # Tags command
    tags_parser = subparsers.add_parser("tags", help="List all available tags")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == "list":
        if args.tag:
            experiments = get_experiments_by_tag(args.tag)
        else:
            experiments = ALL_EXPERIMENTS
        print_experiment_list(experiments, args.detailed)
    
    elif args.command == "run":
        engine = ChaosEngine(
            kubernetes_namespace=getattr(args, 'namespace', 'digital-twin')
        )
        result = asyncio.run(run_experiment(engine, args.experiment, args.dry_run))
        if result:
            print_experiment_result(result)
            if args.output:
                export_results([result], args.output)
    
    elif args.command == "run-tag":
        engine = ChaosEngine(
            kubernetes_namespace=getattr(args, 'namespace', 'digital-twin')
        )
        results = asyncio.run(run_experiments_by_tag(engine, args.tag, args.dry_run))
        if results and args.output:
            export_results(results, args.output)
    
    elif args.command == "tags":
        all_tags = set()
        for exp in ALL_EXPERIMENTS:
            all_tags.update(exp.tags)
        print("\n🏷️  Available Tags:")
        for tag in sorted(all_tags):
            count = len(get_experiments_by_tag(tag))
            print(f"   • {tag} ({count} experiments)")
        print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
