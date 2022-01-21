grid = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]] #Result Grid
grid_pos = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]] #Positional Grid
grid_empty = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2], [3, 3]] #Empty while Grid
empty_pos = []
player_turn = 1
state = 0


def winner(chars):

    if chars[0] is not None and chars[0] != " " and chars[0] is chars[1] and chars[1] is chars[2]:
        if chars[0] == 'X':
            return True
        elif chars[0] == 'O':
            return False
    if chars[0] is not None and chars[0] != " " and chars[0] is chars[3] and chars[3] is chars[6]:
        if chars[0] == 'X':
            return True
        elif chars[0] == 'O':
            return False
    if chars[0] is not None and chars[0] != " " and chars[0] is chars[4] and chars[4] is chars[8]:
        if chars[0] == 'X':
            return True
        elif chars[0] == 'O':
            return False
    if chars[1] is not None and chars[1] != " " and chars[1] is chars[4] and chars[4] is chars[7]:
        if chars[1] == 'X':
            return True
        elif chars[1] == 'O':
            return False
    if chars[2] is not None and chars[2] != " " and chars[2] is chars[4] and chars[4] is chars[6]:
        if chars[2] == 'X':
            return True
        elif chars[2] == 'O':
            return False
    if chars[2] is not None and chars[2] != " " and chars[2] is chars[5] and chars[5] is chars[8]:
        if chars[2] == 'X':
            return True
        elif chars[2] == 'O':
            return False
    if chars[3] is not None and chars[3] != " " and chars[3] is chars[4] and chars[4] is chars[5]:
        if chars[3] == 'X':
            return True
        elif chars[3] == 'O':
            return False
    if chars[6] is not None and chars[6] != " " and chars[6] is chars[7] and chars[7] is chars[8]:
        if chars[6] == 'X':
            return True
        elif chars[6] == 'O':
            return False
    else:
        return None


def confirm_space(pos1, pos2, cor_list):
    for g in cor_list:
        if g[0] == pos1 and g[1] == pos2:
            return cor_list.index(g)


for h, o in enumerate(grid):
    if o != 'X':
        if o != 'O':
            empty_pos.append(h)


for s in empty_pos:
    if grid_empty[s] != 'X':
        grid_empty[s] = ' '
    elif grid_empty[s] != 'O' and grid_empty[s] != 'X':
        grid_empty[s] = ' '


while True:
    print(f"""---------
| {grid_empty[0]} {grid_empty[1]} {grid_empty[2]} |
| {grid_empty[3]} {grid_empty[4]} {grid_empty[5]} |
| {grid_empty[6]} {grid_empty[7]} {grid_empty[8]} |
---------""")
    try:
        x, y = map(int, input("Enter the coordinates: ").split())
        slot = confirm_space(x, y, grid_pos)
        if 4 > x > 0 and 4 > y > 0:
            if grid[slot] == 'X' or grid[slot] == 'O':
                print("This cell is occupied! Choose another one!")
            else:
                if player_turn == 1:
                    for z in grid:
                        if z[0] == x and z[1] == y:
                            pos = grid.index(z)
                            grid[pos] = 'X'
                            grid_empty[pos] = 'X'
                            player_turn -= 1
                elif player_turn == 0:
                    for z in grid:
                        if z[0] == x and z[1] == y:
                            pos = grid.index(z)
                            grid[pos] = 'O'
                            grid_empty[pos] = 'O'
                            player_turn += 1
            if winner(grid):
                state = 1
                break
            elif winner(grid) is False:
                state = 2
                break
            elif winner(grid) is None and grid_empty.count(' ') == 0:
                break
        else:
            print("Coordinates should be from 1 to 3!")
    except ValueError:
        print("You should enter numbers!")


for f in empty_pos:
    if grid[f] != 'X' and grid[f] != 'O':
        grid[f] = ' '
    elif grid[f] != 'O' and grid[f] != 'X':
        grid[f] = ' '

print(f"""---------
| {grid[0]} {grid[1]} {grid[2]} |
| {grid[3]} {grid[4]} {grid[5]} |
| {grid[6]} {grid[7]} {grid[8]} |
---------""")
if state == 1:
    print("X wins")
elif state == 2:
    print("O wins")
else:
    print("Draw")
