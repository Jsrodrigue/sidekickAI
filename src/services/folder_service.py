import os
import asyncio

from src.core.sidekick import Sidekick
from src.utils.path_utils import normalize_path, get_absolute_path


def _make_index_key(path: str) -> str:
    """
    Must match Sidekick's indexing key behavior:
    - directory => absolute path
    - file => FILE::<absolute_path>
    """
    abs_path = get_absolute_path(normalize_path(path))
    if os.path.isfile(abs_path):
        return f"FILE::{abs_path}"
    return abs_path


class FolderService:
    def __init__(self, sidekick: Sidekick):
        self.sidekick = sidekick

    def _validate_path_exists(self, path: str) -> None:
        if not (os.path.isdir(path) or os.path.isfile(path)):
            raise ValueError(f"Path does not exist: {path}")

    async def ensure_indexed(
        self,
        folder: str,
        state,
        chunk_size: int = 600,
        chunk_overlap: int = 20,
    ):
        """
        Ensure that the path is indexed.
        Backward compatible: argument name is still `folder`,
        but it can be a directory OR a file path.

        chunk_size and chunk_overlap control how documents are split into chunks.
        """
        path = normalize_path(folder)
        self._validate_path_exists(path)

        # Track the "current" selection in the user's GLOBAL state
        state.current_directory = path

        # Check if already indexed using the SAME keying as Sidekick
        index_key = _make_index_key(path)

        if not self.sidekick.retrieval_service.has_retriever(index_key):
            # index_path is synchronous -> run in worker thread
            result_msg = await asyncio.to_thread(
                self.sidekick.index_path,
                path,
                False,           # force_reindex=False
                chunk_size,
                chunk_overlap,
                True,            # recursive=True (only matters for dirs)
            )
            print(result_msg)

        return state

    async def reindex_folder(
        self,
        folder: str,
        state,
        chunk_size: int = 600,
        chunk_overlap: int = 20,
    ):
        """
        Force reindexing of the selected path (folder OR file).

        Uses the given chunk_size and chunk_overlap to rebuild
        the vectorstore for this path.
        """
        path = normalize_path(folder)
        self._validate_path_exists(path)

        state.current_directory = path

        result_msg = await asyncio.to_thread(
            self.sidekick.index_path,
            path,
            True,            # force_reindex=True
            chunk_size,
            chunk_overlap,
            True,            # recursive=True
        )
        print(result_msg)
        return state

    async def clear_folder(self, folder: str, state):
        """
        Clear the index of the selected path (folder OR file),
        delete its vectorstore and update the state.
        """
        path = normalize_path(folder)
        self._validate_path_exists(path)

        state.current_directory = path

        result_msg = await asyncio.to_thread(self.sidekick.remove_path, path)
        print(result_msg)
        return state
