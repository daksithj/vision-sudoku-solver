import copy


class Sudoku:

    def __init__(self, puzzle, puzzle_size):

        self.puzzle = puzzle    # The puzzle array

        self.puzzle_size = puzzle_size  # Size of the puzzle

        self.puzzle_history = []        # Keeps track of changes to puzzle to revert when backtracking

        self.solution = None        # The solution the puzzle

        self.filled_count = 0       # The number of filled spaces in the puzzle

        box_dim = puzzle_size ** 0.5
        self.box_dim = int(box_dim)     # The size of a side of a box (section)

        self.all_false = [False] * puzzle_size  # Cell value when no values are allowed

        # Array to store the allowed values for each cell as a boolean array
        self.allowed_values = [[[True for _ in range(puzzle_size)] for _ in range(puzzle_size)]
                               for _ in range(puzzle_size)]

        # Array storing the filled values in each row
        self.row_fillings = [[True for _ in range(puzzle_size)] for _ in range(puzzle_size)]

        # Array storing the filled values in each column
        self.column_fillings = [[True for _ in range(puzzle_size)] for _ in range(puzzle_size)]

        # Array storing the filled values in each block
        self.box_fillings = [[[True for _ in range(puzzle_size)] for _ in range(self.box_dim)]
                             for _ in range(self.box_dim)]

        # Check for solved values and disallow any values
        for idx, x in enumerate(self.puzzle):
            for idy, y in enumerate(x):

                if y > 0:
                    self.filled_count += 1      # Update the filled count
                    self.set_not_allowed(idx, idy)    # Ensure values aren't allowed in these cells
                    self.adjust_for_existing_values(idx, idy, y-1)
                    self.row_fillings[idx][y-1] = False
                    self.column_fillings[idy][y - 1] = False
                    self.box_fillings[int(idx / self.box_dim)][int(idy / self.box_dim)][y - 1] = False

    # Function to disallow any value to a particular cell
    def set_not_allowed(self, x, y):
        self.allowed_values[x][y] = copy.deepcopy(self.all_false)

    # Function to adjust the allowed values for other cells when a number is entered to the solution
    def adjust_for_existing_values(self, x, y, value):

        # Remove value from allowed list in row with the value
        for idy, allow in enumerate(self.allowed_values[x]):
            self.allowed_values[x][idy][value] = False

        # Remove value from allowed list in column with the value
        for idx, allow in enumerate(self.allowed_values):
            self.allowed_values[idx][y][value] = False

        begin_row = x - x % self.box_dim
        begin_column = y - y % self.box_dim

        # Remove value from allowed list in block with the value
        for i in range(begin_row, begin_row + self.box_dim):
            for j in range(begin_column, begin_column + self.box_dim):
                self.allowed_values[i][j][value] = False

    # Function to set the value of a cell and update other allowed values
    def set_value(self, x, y, value):

        self.puzzle[x][y] = value + 1
        self.set_not_allowed(x, y)
        self.adjust_for_existing_values(x, y, value)
        self.puzzle_history.append((x, y))
        self.row_fillings[x][value] = False
        self.column_fillings[y][value] = False
        self.box_fillings[int(x / self.box_dim)][int(y / self.box_dim)][value] = False

    # Logically deduce some answers using the allowed values
    def deduce_values(self):

        # Adding a stopping point for this deduction (stop wasting resources on futile efforts)
        while self.filled_count < (self.puzzle_size ** 2 - 6):

            changes = 0     # Keeps track of how many values were deduced

            # Scan for cells with only a single allowed value and setting that value in the answer

            for x in range(self.puzzle_size):
                for y in range(self.puzzle_size):

                    if self.puzzle[x][y] == 0:
                        true_val = -1

                        for k in range(self.puzzle_size):

                            if self.allowed_values[x][y][k]:

                                if true_val < 0:
                                    true_val = k
                                else:
                                    true_val = - 1
                                    break

                        if true_val >= 0:

                            self.set_value(x, y, true_val)
                            changes += 1

            # Scan rows for unique allowed values in the row and if found they are set in the answer
            for x in range(self.puzzle_size):
                for k in range(self.puzzle_size):

                    if not self.row_fillings[x][k]:
                        continue

                    containing = -1

                    for y in range(self.puzzle_size):

                        if self.allowed_values[x][y][k]:

                            if containing < 0:
                                containing = y
                            else:
                                containing = -1
                                break

                    if containing >= 0:

                        self.set_value(x, containing, k)
                        changes += 1

            # Scan columns for unique allowed values in the columns and if found they are set in the answer
            for x in range(self.puzzle_size):
                for k in range(self.puzzle_size):

                    if not self.column_fillings[x][k]:
                        continue

                    containing = -1

                    for y in range(self.puzzle_size):

                        if self.allowed_values[y][x][k]:

                            if containing < 0:
                                containing = y
                            else:
                                containing = -1
                                break

                    if containing >= 0:

                        self.set_value(containing, x, k)
                        changes += 1

            # Checking blocks for singled out allowed values and if found they are set in the answer
            for sec_x in range(self.puzzle_size//self.box_dim):
                for sec_y in range(self.puzzle_size // self.box_dim):
                    for k in range(self.puzzle_size):

                        if not self.box_fillings[sec_x][sec_y][k]:
                            continue

                        contain_x, contain_y = self.check_block_allowed(sec_x, sec_y, k)

                        if contain_x >= 0:

                            self.set_value(contain_x, contain_y, k)
                            changes += 1

            self.filled_count += changes    # Update the filled count

            # If the changes made by the deduction is less than this threshold there is no point wasting more time on it
            if changes < 2:
                return

    # Function to check if a block has a cell with unique allowed value
    def check_block_allowed(self, sec_x, sec_y, value):

        contain_x = -1
        contain_y = -1

        for x in range(self.box_dim):
            for y in range(self.box_dim):

                box_x = sec_x * self.box_dim + x
                box_y = sec_y * self.box_dim + y

                if self.allowed_values[box_x][box_y][value]:

                    if contain_x < 0:
                        contain_x = box_x
                        contain_y = box_y
                    else:
                        contain_x = -1
                        return contain_x, contain_y

        return contain_x, contain_y

    # Revert changes to puzzle and fillings arrays when back tracking to a recorded point
    def revert_changes(self, index):

        point = len(self.puzzle_history) - 1

        while point >= index:

            x, y = self.puzzle_history[point]
            value = self.puzzle[x][y]

            self.puzzle[x][y] = 0
            point -= 1

            self.row_fillings[x][value - 1] = True
            self.column_fillings[y][value - 1] = True
            self.box_fillings[int(x / self.box_dim)][int(y / self.box_dim)][value - 1] = True

    # Function to check and get any empty space in the puzzle
    def get_space(self):
        lowest_count = 9
        lowest_idx = -1
        lowest_idy = -1
        for idx, x in enumerate(self.puzzle):
            for idy, y in enumerate(x):
                if y == 0:
                    count = 0
                    for k in self.allowed_values[idx][idy]:
                        if k:
                            count += 1
                    if count == 1:
                        return idx, idy
                    if count < lowest_count:
                        lowest_count = count
                        lowest_idx = idx
                        lowest_idy = idy
        if lowest_idx >= 0:
            return lowest_idx, lowest_idy

        return None

    # Recursive function to check for solutions and backtrack if no available solution
    def backtrack(self):

        self.deduce_values()  # Deduce answer from existing allowed values

        position = self.get_space()   # Get an empty cell

        if position is None:
            # If no empty spaces are available, the puzzle is solved and the solution can be set
            self.solution = self.puzzle
            return True
        else:
            x, y = position     # The coordinates of the empty cell

        # Iterate through all possible values for the cell
        for num in range(1, self.puzzle_size + 1):

            # Only try the value out if it is in the allowed values set
            if self.allowed_values[x][y][num-1]:

                # Storing the changes done to puzzle in history
                self.puzzle_history.append((x, y))

                # Storing the index/point in the history to revert to
                history_index = len(self.puzzle_history) - 1

                # Creating a copy of the pallowed values in case of backtracking
                allowed_values_copy = [[[self.allowed_values[i][j][k] for k in range(self.puzzle_size)]
                                        for j in range(self.puzzle_size)] for i in range(self.puzzle_size)]

                filled_count_copy = self.filled_count    # Save filled count in case it backtracks

                self.set_value(x, y, num - 1)

                self.filled_count += 1

                # If successful to find a solution returns true
                if self.backtrack():
                    return True

                # If impossible to find a solution with chosen path, the backups are restored and backtracked
                self.revert_changes(history_index)
                self.allowed_values = allowed_values_copy
                self.filled_count = filled_count_copy

        return False

    def solve(self):
        if self.backtrack():
            return self.solution
        else:
            return None

    def get_solution(self):
        solution = self.solution

        if solution is None:
            solution = self.solve()

        return solution
