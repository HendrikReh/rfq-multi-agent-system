#!/usr/bin/env python3
"""
Migration script for RFQ Multi-Agent System refactoring.

This script helps migrate from the old flat structure to the new
modular architecture following multi-agent best practices.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import typer
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

console = Console()
app = typer.Typer(help="RFQ System Migration Tool")

# Migration mapping: (old_path, new_path, description)
MIGRATION_MAP: List[Tuple[str, str, str]] = [
    # Core agents
    ("agents/rfq_parser.py", "src/rfq_system/agents/core/rfq_parser.py", "RFQ Parser Agent"),
    ("agents/customer_intent_agent.py", "src/rfq_system/agents/core/customer_intent.py", "Customer Intent Agent"),
    ("agents/conversation_state_agent.py", "src/rfq_system/agents/core/conversation_state.py", "Conversation State Agent"),
    ("agents/interaction_decision_agent.py", "src/rfq_system/agents/core/interaction_decision.py", "Interaction Decision Agent"),
    ("agents/question_generation_agent.py", "src/rfq_system/agents/core/question_generation.py", "Question Generation Agent"),
    ("agents/pricing_strategy_agent.py", "src/rfq_system/agents/core/pricing_strategy.py", "Pricing Strategy Agent"),
    ("agents/customer_response_agent.py", "src/rfq_system/agents/core/customer_response.py", "Customer Response Agent"),
    
    # Specialized agents
    ("agents/competitive_intelligence_agent.py", "src/rfq_system/agents/specialized/competitive_intelligence.py", "Competitive Intelligence Agent"),
    ("agents/risk_assessment_agent.py", "src/rfq_system/agents/specialized/risk_assessment.py", "Risk Assessment Agent"),
    ("agents/contract_terms_agent.py", "src/rfq_system/agents/specialized/contract_terms.py", "Contract Terms Agent"),
    ("agents/proposal_writer_agent.py", "src/rfq_system/agents/specialized/proposal_writer.py", "Proposal Writer Agent"),
    
    # Evaluation agents
    ("agents/evaluation_intelligence_agent.py", "src/rfq_system/agents/evaluation/performance_monitor.py", "Performance Monitor Agent"),
    
    # Orchestration
    ("agents/rfq_orchestrator.py", "src/rfq_system/orchestration/coordinators/sequential.py", "Sequential Coordinator"),
    ("agents/enhanced_orchestrator.py", "src/rfq_system/orchestration/coordinators/graph_based.py", "Graph-based Coordinator"),
    ("agents/integration_framework.py", "src/rfq_system/orchestration/strategies/delegation.py", "Delegation Strategy"),
    
    # Models
    ("agents/models.py", "src/rfq_system/core/models/rfq.py", "Core RFQ Models"),
    
    # Utilities
    ("agents/utils.py", "src/rfq_system/utils/helpers.py", "Utility Functions"),
    ("agents/scenario_recorder.py", "src/rfq_system/monitoring/metrics/performance.py", "Performance Metrics"),
    
    # Tests (to be reorganized)
    ("test_enhanced_agents.py", "tests/unit/test_agents/test_core_agents.py", "Core Agent Tests"),
    ("test_evaluations.py", "tests/unit/test_agents/test_evaluation.py", "Evaluation Tests"),
    ("test_performance.py", "tests/performance/test_load.py", "Performance Tests"),
    ("test_scenario_recording.py", "tests/integration/test_monitoring.py", "Monitoring Tests"),
    
    # Demo files (to examples)
    ("demo_complete_flow.py", "examples/basic_usage.py", "Basic Usage Example"),
    ("demo_integrated_system.py", "examples/advanced_orchestration.py", "Advanced Orchestration Example"),
    ("demonstrate_model_logic.py", "examples/model_showcase.py", "Model Logic Example"),
]

BACKUP_DIR = "backup_old_structure"


def create_backup(source_dir: Path) -> Path:
    """Create a backup of the current structure."""
    backup_path = source_dir / BACKUP_DIR
    
    if backup_path.exists():
        console.print(f"[yellow]Backup directory already exists: {backup_path}[/yellow]")
        if not typer.confirm("Overwrite existing backup?"):
            console.print("[red]Migration cancelled[/red]")
            raise typer.Exit(1)
        shutil.rmtree(backup_path)
    
    console.print(f"[blue]Creating backup at: {backup_path}[/blue]")
    
    # Create backup directory
    backup_path.mkdir(exist_ok=True)
    
    # Copy important files to backup
    files_to_backup = [
        "agents/",
        "*.py",
        "*.md",
        "*.toml",
        "*.ini",
        "*.lock"
    ]
    
    for pattern in files_to_backup:
        for file_path in source_dir.glob(pattern):
            if file_path.is_file():
                dest = backup_path / file_path.name
                shutil.copy2(file_path, dest)
            elif file_path.is_dir() and file_path.name != BACKUP_DIR:
                dest = backup_path / file_path.name
                shutil.copytree(file_path, dest, dirs_exist_ok=True)
    
    console.print(f"[green]Backup created successfully[/green]")
    return backup_path


def migrate_file(old_path: Path, new_path: Path, description: str) -> bool:
    """Migrate a single file from old location to new location."""
    if not old_path.exists():
        console.print(f"[yellow]Skipping {description}: {old_path} not found[/yellow]")
        return False
    
    # Create destination directory
    new_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy file to new location
        shutil.copy2(old_path, new_path)
        console.print(f"[green]✓[/green] Migrated {description}")
        return True
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to migrate {description}: {e}")
        return False


def update_imports_in_file(file_path: Path) -> None:
    """Update import statements in a migrated file."""
    if not file_path.exists() or file_path.suffix != '.py':
        return
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Common import updates
        import_updates = {
            "from agents.": "from rfq_system.agents.",
            "from .models": "from rfq_system.core.models.rfq",
            "from .utils": "from rfq_system.utils.helpers",
            "import agents.": "import rfq_system.agents.",
        }
        
        updated = False
        for old_import, new_import in import_updates.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                updated = True
        
        if updated:
            with open(file_path, 'w') as f:
                f.write(content)
            console.print(f"[blue]Updated imports in {file_path.name}[/blue]")
    
    except Exception as e:
        console.print(f"[yellow]Warning: Could not update imports in {file_path}: {e}[/yellow]")


@app.command()
def migrate(
    source_dir: str = typer.Argument(".", help="Source directory (current project root)"),
    create_backup_flag: bool = typer.Option(True, "--backup/--no-backup", help="Create backup before migration"),
    update_imports: bool = typer.Option(True, "--update-imports/--no-update-imports", help="Update import statements"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be migrated without doing it")
):
    """
    Migrate the RFQ system from old structure to new modular architecture.
    """
    source_path = Path(source_dir).resolve()
    
    console.print(f"[bold]RFQ System Migration Tool[/bold]")
    console.print(f"Source directory: {source_path}")
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
    
    # Validate source directory
    if not source_path.exists():
        console.print(f"[red]Error: Source directory does not exist: {source_path}[/red]")
        raise typer.Exit(1)
    
    # Check if this looks like the RFQ project
    if not (source_path / "agents").exists():
        console.print("[yellow]Warning: 'agents' directory not found. Is this the correct project directory?[/yellow]")
        if not typer.confirm("Continue anyway?"):
            raise typer.Exit(1)
    
    # Create backup if requested
    backup_path = None
    if create_backup_flag and not dry_run:
        backup_path = create_backup(source_path)
    
    # Show migration plan
    table = Table(title="Migration Plan")
    table.add_column("Source", style="cyan")
    table.add_column("Destination", style="green")
    table.add_column("Description", style="yellow")
    table.add_column("Status", style="white")
    
    migration_results = []
    
    with Progress() as progress:
        task = progress.add_task("Migrating files...", total=len(MIGRATION_MAP))
        
        for old_rel_path, new_rel_path, description in MIGRATION_MAP:
            old_path = source_path / old_rel_path
            new_path = source_path / new_rel_path
            
            if dry_run:
                status = "✓ Would migrate" if old_path.exists() else "⚠ Not found"
                table.add_row(old_rel_path, new_rel_path, description, status)
            else:
                success = migrate_file(old_path, new_path, description)
                migration_results.append((new_path, success))
                
                if success and update_imports:
                    update_imports_in_file(new_path)
                
                status = "✓ Migrated" if success else "✗ Failed"
                table.add_row(old_rel_path, new_rel_path, description, status)
            
            progress.advance(task)
    
    console.print(table)
    
    if not dry_run:
        # Summary
        successful = sum(1 for _, success in migration_results if success)
        total = len(migration_results)
        
        console.print(f"\n[bold]Migration Summary:[/bold]")
        console.print(f"Successfully migrated: {successful}/{total} files")
        
        if backup_path:
            console.print(f"Backup created at: {backup_path}")
        
        if successful > 0:
            console.print("\n[green]Next steps:[/green]")
            console.print("1. Review migrated files for any manual adjustments needed")
            console.print("2. Update import statements in remaining files")
            console.print("3. Run tests to ensure everything works")
            console.print("4. Update documentation and README")
            console.print("5. Consider removing old files after verification")


@app.command()
def status(source_dir: str = typer.Argument(".", help="Source directory")):
    """
    Show migration status - what files exist and where they would be migrated.
    """
    source_path = Path(source_dir).resolve()
    
    table = Table(title="Migration Status")
    table.add_column("Source File", style="cyan")
    table.add_column("Exists", style="white")
    table.add_column("Destination", style="green")
    table.add_column("Dest Exists", style="white")
    
    for old_rel_path, new_rel_path, description in MIGRATION_MAP:
        old_path = source_path / old_rel_path
        new_path = source_path / new_rel_path
        
        old_exists = "✓" if old_path.exists() else "✗"
        new_exists = "✓" if new_path.exists() else "✗"
        
        table.add_row(old_rel_path, old_exists, new_rel_path, new_exists)
    
    console.print(table)


@app.command()
def cleanup(
    source_dir: str = typer.Argument(".", help="Source directory"),
    force: bool = typer.Option(False, "--force", help="Force cleanup without confirmation")
):
    """
    Clean up old files after successful migration.
    """
    source_path = Path(source_dir).resolve()
    
    # Find old files that have been successfully migrated
    old_files_to_remove = []
    
    for old_rel_path, new_rel_path, description in MIGRATION_MAP:
        old_path = source_path / old_rel_path
        new_path = source_path / new_rel_path
        
        if old_path.exists() and new_path.exists():
            old_files_to_remove.append((old_path, description))
    
    if not old_files_to_remove:
        console.print("[green]No old files to clean up[/green]")
        return
    
    console.print(f"[yellow]Found {len(old_files_to_remove)} old files to clean up:[/yellow]")
    for old_path, description in old_files_to_remove:
        console.print(f"  - {old_path} ({description})")
    
    if not force:
        if not typer.confirm("\nProceed with cleanup?"):
            console.print("Cleanup cancelled")
            return
    
    # Remove old files
    for old_path, description in old_files_to_remove:
        try:
            old_path.unlink()
            console.print(f"[green]✓[/green] Removed {old_path}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to remove {old_path}: {e}")
    
    # Remove empty directories
    old_agents_dir = source_path / "agents"
    if old_agents_dir.exists() and not any(old_agents_dir.iterdir()):
        try:
            old_agents_dir.rmdir()
            console.print(f"[green]✓[/green] Removed empty directory: {old_agents_dir}")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not remove {old_agents_dir}: {e}[/yellow]")


@app.command()
def migrate_tests(
    source_dir: str = typer.Argument(".", help="Source directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be migrated without doing it")
):
    """Migrate test files to proper test directory structure."""
    console.print("🧪 Migrating test files...")
    
    source_path = Path(source_dir).resolve()
    
    # Test file mappings: (source_file, target_directory, new_name)
    test_migrations = [
        # Unit tests for agents
        ("test_enhanced_agents.py", "tests/unit/test_agents", "test_enhanced_agents.py"),
        ("test_model_assignment.py", "tests/unit/test_agents", "test_model_assignment.py"),
        
        # Integration tests
        ("test_evaluations.py", "tests/integration", "test_evaluations.py"),
        ("test_verification.py", "tests/integration", "test_verification.py"),
        ("test_scenario_recording.py", "tests/integration", "test_scenario_recording.py"),
        ("test_simple.py", "tests/integration", "test_simple.py"),
        
        # Performance tests
        ("test_performance.py", "tests/performance", "test_performance.py"),
        
        # Test utilities and runners
        ("run_all_tests.py", "tests", "run_all_tests.py"),
        
        # Demo/example files (move to examples)
        ("demo_complete_flow.py", "examples", "demo_complete_flow.py"),
        ("demo_integrated_system.py", "examples", "demo_integrated_system.py"),
        ("demonstrate_model_logic.py", "examples", "demonstrate_model_logic.py"),
        ("view_scenarios.py", "examples", "view_scenarios.py"),
        ("show_model_config.py", "examples", "show_model_config.py"),
    ]
    
    for source_file, target_dir, new_name in test_migrations:
        source_path_file = source_path / source_file
        target_path = source_path / target_dir / new_name
        
        if source_path_file.exists():
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            if dry_run:
                console.print(f"  Would move: {source_path_file} -> {target_path}")
            else:
                # Move the file
                shutil.move(str(source_path_file), str(target_path))
                console.print(f"  [green]✅ Moved: {source_path_file} -> {target_path}[/green]")
        else:
            console.print(f"  [yellow]⚠️  Source not found: {source_path_file}[/yellow]")
    
    # Create test configuration files
    test_configs = [
        ("tests/__init__.py", "# Test package"),
        ("tests/conftest.py", '''"""Test configuration and fixtures."""
import pytest
from pydantic_ai import models

# Disable model requests globally for tests
models.ALLOW_MODEL_REQUESTS = False

@pytest.fixture(autouse=True)
def disable_model_requests():
    """Ensure model requests are disabled for all tests."""
    original = models.ALLOW_MODEL_REQUESTS
    models.ALLOW_MODEL_REQUESTS = False
    yield
    models.ALLOW_MODEL_REQUESTS = original
'''),
        ("tests/unit/__init__.py", "# Unit tests package"),
        ("tests/integration/__init__.py", "# Integration tests package"), 
        ("tests/performance/__init__.py", "# Performance tests package"),
        ("tests/fixtures/__init__.py", "# Test fixtures package"),
    ]
    
    for file_path, content in test_configs:
        path = source_path / file_path
        if not path.exists():
            if dry_run:
                console.print(f"  Would create: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                console.print(f"  [green]✅ Created: {path}[/green]")
        else:
            console.print(f"  [blue]Already exists: {path}[/blue]")


if __name__ == "__main__":
    app() 