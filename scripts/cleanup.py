"""
RAG Storage Cleanup Script

Safely removes all indexed data from PostgreSQL storage and local artifacts.

Usage:
    python scripts/cleanup.py [options]

Options:
    --drop-tables       Drop tables completely (requires reinitialization)
    --dry-run           Show what would be deleted without executing
    --include-local     Also remove local rag_storage directory
    --include-artifacts Also remove output/ and parsing artifacts
    --all               Clean everything (DB + local + artifacts)

Examples:
    python scripts/cleanup.py                    # Truncate DB tables only
    python scripts/cleanup.py --dry-run          # Preview what would be deleted
    python scripts/cleanup.py --include-artifacts  # DB + output artifacts
    python scripts/cleanup.py --all              # Full cleanup
"""

import os
import asyncio
import argparse

from dotenv import load_dotenv

load_dotenv()

# PostgreSQL configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_USER = os.getenv("POSTGRES_USER", "pgvector")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pgvector")
POSTGRES_DATABASE = os.getenv("POSTGRES_DATABASE", "ezis_rag")

# LightRAG tables
LIGHTRAG_TABLES = [
    "lightrag_doc_chunks",
    "lightrag_doc_full",
    "lightrag_doc_status",
    "lightrag_entity_chunks",
    "lightrag_full_entities",
    "lightrag_full_relations",
    "lightrag_llm_cache",
    "lightrag_relation_chunks",
    "lightrag_vdb_chunks",
    "lightrag_vdb_entity",
    "lightrag_vdb_relation",
]


async def get_table_counts(conn) -> dict:
    """Get row counts for all tables"""
    counts = {}
    for table in LIGHTRAG_TABLES:
        try:
            result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
            counts[table] = result
        except Exception:
            counts[table] = "N/A"
    return counts


async def truncate_tables(conn, dry_run: bool = False):
    """Truncate all LightRAG tables (keep structure)"""
    print("\n[TRUNCATE] Clearing table data...")

    for table in LIGHTRAG_TABLES:
        try:
            if dry_run:
                print(f"  [DRY-RUN] Would truncate: {table}")
            else:
                await conn.execute(f"TRUNCATE TABLE {table} CASCADE")
                print(f"  Truncated: {table}")
        except Exception as e:
            print(f"  Skip (not exists): {table}")


async def drop_tables(conn, dry_run: bool = False):
    """Drop all LightRAG tables completely"""
    print("\n[DROP] Removing tables...")

    for table in LIGHTRAG_TABLES:
        try:
            if dry_run:
                print(f"  [DRY-RUN] Would drop: {table}")
            else:
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                print(f"  Dropped: {table}")
        except Exception as e:
            print(f"  Error dropping {table}: {e}")


async def cleanup_age_graphs(conn, dry_run: bool = False):
    """Remove Apache AGE graphs"""
    print("\n[AGE] Cleaning up graph data...")

    try:
        # Get list of graphs
        graphs = await conn.fetch("""
            SELECT name FROM ag_catalog.ag_graph
            WHERE name != 'ag_catalog'
        """)

        for graph in graphs:
            graph_name = graph['name']
            if dry_run:
                print(f"  [DRY-RUN] Would drop graph: {graph_name}")
            else:
                await conn.execute(f"SELECT * FROM ag_catalog.drop_graph('{graph_name}', true)")
                print(f"  Dropped graph: {graph_name}")

        if not graphs:
            print("  No graphs to clean")

    except Exception as e:
        print(f"  Error cleaning graphs: {e}")


def cleanup_local_storage(dry_run: bool = False):
    """Remove local rag_storage directory"""
    import shutil

    storage_dir = "./rag_storage"
    if os.path.exists(storage_dir):
        if dry_run:
            print(f"\n[LOCAL] Would remove: {storage_dir}")
        else:
            print(f"\n[LOCAL] Removing {storage_dir}...")
            shutil.rmtree(storage_dir)
            print(f"  Removed: {storage_dir}")
    else:
        print(f"\n[LOCAL] {storage_dir} does not exist")


def cleanup_output_artifacts(dry_run: bool = False):
    """Remove output directory and parsing artifacts"""
    import shutil

    # Directories to clean
    artifact_dirs = [
        "./output",           # MinerU/Docling parsing output
        "./rag_storage",      # LightRAG local storage (if not using PG)
    ]

    print("\n[ARTIFACTS] Cleaning output and parsing artifacts...")

    for dir_path in artifact_dirs:
        if os.path.exists(dir_path):
            if dry_run:
                # Count files
                file_count = sum(len(files) for _, _, files in os.walk(dir_path))
                print(f"  [DRY-RUN] Would remove: {dir_path} ({file_count} files)")
            else:
                shutil.rmtree(dir_path)
                print(f"  Removed: {dir_path}")
        else:
            print(f"  Skip (not exists): {dir_path}")


async def main():
    parser = argparse.ArgumentParser(description="Cleanup RAG storage")
    parser.add_argument("--drop-tables", action="store_true",
                        help="Drop tables completely")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted")
    parser.add_argument("--include-local", action="store_true",
                        help="Also remove local rag_storage directory")
    parser.add_argument("--include-artifacts", action="store_true",
                        help="Also remove output/ and parsing artifacts")
    parser.add_argument("--all", action="store_true",
                        help="Clean everything (DB + local + artifacts)")
    args = parser.parse_args()

    # --all enables all cleanup options
    if args.all:
        args.include_local = True
        args.include_artifacts = True

    print("=" * 60)
    print("RAG Storage Cleanup")
    print("=" * 60)
    print(f"\nPostgreSQL: {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***")

    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=POSTGRES_HOST,
            port=int(POSTGRES_PORT),
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DATABASE,
        )

        # Show current state
        print("\n[CURRENT STATE]")
        counts = await get_table_counts(conn)
        total = 0
        for table, count in counts.items():
            if isinstance(count, int):
                total += count
                print(f"  {table}: {count} rows")
            else:
                print(f"  {table}: {count}")
        print(f"  Total: {total} rows")

        if total == 0 and not args.drop_tables:
            print("\nNo data to clean.")
            await conn.close()
            return

        # Confirm
        if not args.dry_run:
            confirm = input("\nProceed with cleanup? [y/N]: ")
            if confirm.lower() != 'y':
                print("Cancelled.")
                await conn.close()
                return

        # Cleanup AGE graphs first
        await cleanup_age_graphs(conn, args.dry_run)

        # Truncate or drop tables
        if args.drop_tables:
            await drop_tables(conn, args.dry_run)
        else:
            await truncate_tables(conn, args.dry_run)

        # Cleanup local storage
        if args.include_local:
            cleanup_local_storage(args.dry_run)

        # Cleanup output artifacts
        if args.include_artifacts:
            cleanup_output_artifacts(args.dry_run)

        await conn.close()

        print("\n" + "=" * 60)
        print("Cleanup completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
