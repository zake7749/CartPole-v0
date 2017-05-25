import gym
import os
import random
import numpy as np

from functools import reduce
from keras.layers import Dense, Activation, Input, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.models import Model

random_seed = 1024

np_random = np.random.RandomState(random_seed)
random.seed(random_seed)

def random_game(game_step):

    '''
    Playing the game by random process.

    Args:
        - game_step : the game step that the env can takes.

    Return:
        - score: It is the final score whne game is done.
        - obseravations: record the observations during this random process.
        - actions: record the actions we taken during this random process.
    '''
    env = gym.make('CartPole-v0')
    env.reset()
    env.seed(random_seed)

    prev_obseravtion = None
    obseravations = []
    actions = []
    score = 0

    for _ in range(game_step):

        random_action = random.randrange(0, 2)

        if prev_obseravtion is not None:
            obseravations.append(prev_obseravtion)

            if random_action == 0:
                actions.append([1, 0])
            else:
                actions.append([0, 1])

        observation, reward, done, info = env.step(random_action)

        if done:
            break

        prev_obseravtion = observation
        score += reward

    return score, obseravations, actions


def generate_training_data(score_bound, game_nb, game_step, generate_new_data=False):

    '''
    Generate the acceptable by playing the game randomly

    Args:
       - score_bound: a bound that block the game memory which has a too low score into training data.
       - game_nb: the number of acceptable game.
       - game_step : the game step that the env can takes.

    :return: observations, actions, mean_score
    '''

    if os.path.exists('training_x') and os.path.exists('training_y') and not generate_new_data:
        observations = np.load("training_x")
        actions = np.load("training_y")

    else:
        observations = []
        actions = []
        scores = []  # to give a overview performance of random processes
        tried = 0
        accepted = 0

        while accepted < game_nb:

            score, o, a = random_game(game_step)
            tried += 1

            if score > score_bound:
                accepted += 1
                scores.append(score)
                observations = observations + o
                actions = actions + a

        observations = np.array(observations)
        actions = np.array(actions)

        np.save('training_x', observations)
        np.save('training_y', actions)

    print("Has played {} games".format(tried))
    print("Get {} acceptable game results".format(len(observations)))
    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))

    return observations, actions, np.mean(scores)


def fully_connected_neural_network(input_size, output_size):

    inputs = Input(shape=(input_size,))

    network = Dense(256)(inputs)
    network = Activation('relu')(network)

    network = Dense(512)(network)
    network = Activation('relu')(network)

    network = Dense(512)(network)
    network = Activation('relu')(network)

    network = Dense(256)(network)
    network = Activation('relu')(network)

    network = Dense(output_size)(network)
    prediction = Activation('softmax')(network)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train(observations, actions, model, nb_epoch=3):

    '''
    Train the DNN.
    '''

    model.fit(x=observations, y=actions, nb_epoch=int(nb_epoch))
    model.save('CartPole-v0')

    return model


def evaluate(model, epoch=1):

    env = gym.make('CartPole-v0')
    env.reset()
    env.seed(random_seed)

    scores = []

    observations = []
    actions = []

    for _ in range(epoch):
        done = False
        prev_observation = None
        score = 0
        env.reset()
        while not done:

            env.render()

            if prev_observation is None or np_random.uniform(0, 1) > 0.9:  # for variety
                action = random.randrange(0, 2)
            else:
                po = prev_observation.reshape(1, 4)
                action = np.argmax(model.predict(po))
                observations.append(prev_observation.reshape(4))
                if action == 0:
                    actions.append([1, 0])
                else:
                    actions.append([0, 1])

            observation, reward, done, info = env.step(action)
            prev_observation = observation
            score += reward

        scores.append(score)

    observations = np.array(observations)
    actions = np.array(actions)

    print("Average score: {}".format(np.mean(scores)))
    print("Median score: {}".format(np.median(scores)))
    return observations, actions, np.mean(scores)


def learning_by_learning(learning_generation, step, model):

    for epoch in range(learning_generation):

        print("Generation {}".format(epoch))
        observations, actions, mean_score = evaluate(model, 1)

        if mean_score > step:
            if step < 180:
                step += mean_score / 10
            train(observations, actions, model, 1)

def main():

    observations, actions, mean_score = generate_training_data(40, 20, 200, False)
    model = fully_connected_neural_network(4, 2)
    model = train(observations, actions, model, 1)
    learning_by_learning(100, mean_score, model)

if __name__ == "__main__":
    main()