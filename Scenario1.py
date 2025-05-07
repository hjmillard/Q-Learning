from FourRooms import FourRooms
import numpy as np

Q_values = np.ones((13, 13, 4)) * -1    # (rows,cols,actions) of the environment
epsilon = 0.9
gamma = 0.9
alpha = 0.9

def choose_action(current_row, current_col, epsilon, Q_values):
    if np.random.random() > epsilon:
        return np.argmax(Q_values[current_row, current_col])
    else: #choose a random action
        return np.random.randint(4)

def main():

    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple')

    # Training
    position = (0,0)
    for episode in range(1000):
        current_row, current_col = fourRoomsObj.getPosition()   # starting position for episode
        position = fourRoomsObj.getPosition()
        count = 0
        while not fourRoomsObj.isTerminal():
            count += 1
            action = choose_action(current_row, current_col, epsilon, Q_values)
            old_row, old_col = current_row, current_col
            old_position = position
            cell_type, position, num_packages, is_terminal = fourRoomsObj.takeAction(action)    # take a step
            old_q = Q_values[old_row][old_col]
            if cell_type == 0:
                reward = -1
            else:
                reward = 100
            TD = reward + (gamma * np.max(Q_values[position[0], position[1], action])) - old_q  # calculate the quality of the step
            new_q = old_q + (alpha * TD)
            Q_values[old_row][old_col] = new_q  # update the quality table
            if count == 1000:
                break
    fourRoomsObj.showPath(-1)
    fourRoomsObj.newEpoch()


if __name__ == "__main__":
    main()
