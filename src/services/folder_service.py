import os
import asyncio
from src.core.sidekick import Sidekick

class FolderService:
    def __init__(self, sidekick: Sidekick):
        self.sidekick = sidekick

    async def ensure_indexed(self, folder: str, state):
        """
        Ensure that the folder is indexed.
        Returns the updated state.
        """
        if not os.path.isdir(folder):
            raise ValueError(f"Folder does not exist: {folder}")

        state.current_directory = folder

        # Check if already indexed
        folder_key = os.path.abspath(folder)
        if not self.sidekick.retrieval_service.has_retriever(folder_key):
            # Index using Sidekick method
            # ⚠️ index_directory is synchronous, use asyncio.to_thread to avoid blocking
            result_msg = await asyncio.to_thread(self.sidekick.index_directory, folder)
            print(result_msg)

        return state

    async def reindex_folder(self, folder: str, state):
        """
        Reindex the selected folder.
        """
        if not state.current_directory:
            raise ValueError("No folder selected")
        
        folder = state.current_directory
        result_msg = await asyncio.to_thread(self.sidekick.index_directory, folder, True)
        print(result_msg)
        return state

    async def clear_folder(self, folder: str, state):
        """
        Clear the index of the selected folder.
        """
        if not state.current_directory:
            raise ValueError("No folder selected")
        
        folder = state.current_directory
        result_msg = await asyncio.to_thread(self.sidekick.remove_directory, folder)
        print(result_msg)
        return state
