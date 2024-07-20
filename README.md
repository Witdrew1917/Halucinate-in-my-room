# Halucinate-in-my-room
Project Idea: Train a Nerf of my living room and wrap the model to a camera
positioning engine (game engine), then querry images based on user inputs.
At the same time, laverage a generative model (stable diffusion) such that the
model may halucinate once the bounadries of the recorded data of the room has
been reached.

Task list:

NeRF
 - Record Data []
 - Prepare Data []
 - Code backbone []
 - Train, tune etc. []
 - Wrap the model in a game loop with appropriate inputs []

Stable diffusion
 - Read up on stable diffusion lol 


 Updated list:
 - Get the test set to work []
    - Figure out how to produce the dataset made out of rays []
    - Code the model []
    - Train []
 - Create a game loop that loads the model and allows for navigation within
    a window []
 - Look at Colmap, and add to pipline, or code something yourself []
 - Combine colmap with the test set model and create a pipline that:
    - Inputs the set of images []
    - Creates the data set of rays for the scene []
        - This code should also be able to take an arbitrary camera pose and
          turn it into a set of rays []
    - Train the model []
 - Extend the game loop to use bits of this code to generate rays []

