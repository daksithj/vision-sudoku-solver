import copy

class sudokuSolver3by3:
    def __init__(self, dim, bord):
        self.dim = dim
        self.expandedNodes = 0
        self.board = bord
        self.rv = self.getRemainingValues()
        # print(self.rv)

    def printValue(self):
        print(self.board)
        # return string

    def getDomainLength(self, lst):
        if -15 in lst or lst == []:
            return 10
        else:
            return len(lst)

    def getDomain(self, row, col):
        RVCell = [i for i in range(1, self.dim + 1)]
        for i in range(self.dim):
            if self.board[row][i] != 0:
                if self.board[row][i] in RVCell:
                    RVCell.remove(self.board[row][i])

        for i in range(self.dim):
            if self.board[i][col] != 0:
                if self.board[i][col] in RVCell:
                    RVCell.remove(self.board[i][col])

        boxRow = row - row % 3
        boxCol = col - col % 3
        for i in range(3):
            for j in range(3):
                if self.board[boxRow + i][boxCol + j] != 0:
                    if self.board[boxRow + i][boxCol + j] in RVCell:
                        RVCell.remove(self.board[boxRow + i][boxCol + j])
        return RVCell

    def getNextMRVRowCol(self):
        # print(self.rv);
        rvMap = list(map(self.getDomainLength, self.rv))
        minimum = min(rvMap)
        if minimum == 10:
            return (-1, -1)
        index = rvMap.index(minimum)
        # print("_",index)
        return (index // 9, index % 9)

    def getRemainingValues(self):
        RV = []
        for row in range(self.dim):
            for col in range(self.dim):
                if self.board[row][col] != 0:
                    RV.append([-15])
                else:
                    RV.append(self.getDomain(row, col))
        return RV

    def isEmptyDomainProduced(self, row, col, choice):
        element = self.rv.pop(row * 9 + col)
        if [] in self.rv:
            self.rv.insert(row * 9 + col, element)
            return True
        else:
            self.rv.insert(row * 9 + col, element)
            return False

    def solve(self):
        self.solveCSPFH()
        return self.board;

    def solveCSPFH(self):
        location = self.getNextMRVRowCol()
        # print("------")
        # print(location[0], location[1])
        if location[0] == -1:
            return True
        else:
            self.expandedNodes += 1
            # rv = self.getRemainingValues()
            row = location[0]
            col = location[1]
            for choice in self.rv[row * 9 + col]:
                # print("Before")
                # print(self.rv[row * 9 + col])
                # print("choice->", choice)
                # choice_str = str(choice)
                self.board[row][col] = choice
                # print(self.rv)
                cpy = copy.deepcopy(self.rv)
                self.rv = self.getRemainingValues()

                if not self.isEmptyDomainProduced(row, col, choice):
                    if self.solveCSPFH():
                        return True
                self.board[row][col] = 0
                self.rv = cpy
                # print("After")
                # print(self.rv[row * 9 + col])

            return False