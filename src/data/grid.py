import re
from pathlib import Path 

class RamanGrid:
    def __init__(self, path, x, y):
        self.path = Path(path).expanduser() 
        self.x = x
        self.y = y

    def _get_file_number(self, path):
        """Gets the file number of the corresponding file. 
            eg. ".../FullGrid[1](-18000, -320, -14675)" -> 1

        Args:
            path (Path): Path to the file.

        Returns:
            int : File number of the corresponding file. 
                Returns None if no file number found.
        """
        match = re.search(r'\[(\d+)\]', path.name)
        return int(match.group(1)) if match else None

    def get_sorted_files(self):
         # extract .txt files and store as list
        files = list(self.path.glob('*.txt'))
        # if no files found
        if not files:
            raise FileNotFoundError(f"No .txt files found in {self.path}")
        # sort the files in the order of their file number
        files.sort(key=self._get_file_number)
        return files
    
    def _step(self, cur_x, cur_y, step):
        """_summary_

        Args:
            cur_x (_type_): _description_
            cur_y (_type_): _description_
            step (_type_): _description_

        Returns:
            _type_: _description_
        """
        if cur_y == self.y - 1 and step != -1: # on right boundary
            cur_x -= 1 # step up
            step *= -1 # flip step direction
        elif cur_y == 0 and step != 1: # on left boundary
            cur_x -= 1 # step up
            step *= -1 # flip step direction
        else: # not on boundary
            cur_y += step # step 
        return cur_x, cur_y, step

    def traverse(self):
        """Generator to yield one position at each call.

        Yields:
            _type_: _description_
        """
        files = self.get_sorted_files()
        # initial position and step direction
        cur_x, cur_y, step = self.x - 1, 0, 1
        for file in files:
            yield file, cur_x, cur_y
            cur_x, cur_y, step = self._step(cur_x, cur_y, step)