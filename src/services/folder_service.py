import os
import asyncio

from src.core.sidekick import Sidekick


class FolderService:
    def __init__(self, sidekick: Sidekick):
        self.sidekick = sidekick

    async def ensure_indexed(
        self,
        folder: str,
        state,
        chunk_size: int = 600,
        chunk_overlap: int = 20,
    ):
        """
        Ensure that the folder is indexed.
        Returns the updated state.

        chunk_size and chunk_overlap control how documents are split into chunks.
        """
        if not os.path.isdir(folder):
            raise ValueError(f"Folder does not exist: {folder}")

        # Track the "current" directory in the user's GLOBAL state
        state.current_directory = folder

        # Check if already indexed
        folder_key = os.path.abspath(folder)
        if not self.sidekick.retrieval_service.has_retriever(folder_key):
            # Index using Sidekick method
            # index_directory is synchronous -> run in a worker thread
            result_msg = await asyncio.to_thread(
                self.sidekick.index_directory,
                folder,
                False,            # force_reindex=False
                chunk_size,
                chunk_overlap,
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
        Force reindexing of the selected folder.

        Uses the given chunk_size and chunk_overlap to rebuild
        the vectorstore for this folder.
        """
        if not os.path.isdir(folder):
            raise ValueError(f"Folder does not exist: {folder}")

        # Update current directory in state
        state.current_directory = folder

        result_msg = await asyncio.to_thread(
            self.sidekick.index_directory,
            folder,
            True,              # force_reindex=True
            chunk_size,
            chunk_overlap,
        )
        print(result_msg)
        return state

    async def clear_folder(self, folder: str, state):
        """
        Clear the index of the selected folder (delete its vectorstore)
        and update the state.
        """
        if not os.path.isdir(folder):
            raise ValueError(f"Folder does not exist: {folder}")

        # Keep track of which folder was last cleared
        state.current_directory = folder

        result_msg = await asyncio.to_thread(self.sidekick.remove_directory, folder)
        print(result_msg)
        return state
