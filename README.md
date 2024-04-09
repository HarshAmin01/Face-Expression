# Face-Expression Prediction and Face Generation

This Deep Learning models predicts the human face expressions based on the image provided in the input section.
Also there are generative models which tries to generates similar images based on the provided input.

![image](https://github.com/HarshAmin01/Face-Expression/assets/101825662/d8fee108-54f2-4447-8c20-f7a56d841034)

## Predictive Models:
- Predictive models are trained on face expression prediction part. The basic work flow is that it provides a prediction of a face expression of given images.

- There are three predictive models
  1. Transfer learnt VGGNET19 with additional layers
  2. Transfer learnt Efficient Net with unfreezing some layers
  3. Convolution Neural Network
![image](https://github.com/HarshAmin01/Face-Expression/assets/101825662/5f42c625-bb1a-4405-be09-00f3dc7dae51)

- In the file upload section user can upload an image of a human face and then user has to select a particular model on which they want a prediction.
- After selecting a model when user clicks on Detect image button the selected model provides an output.

 ![Screenshot (15)](https://github.com/HarshAmin01/Face-Expression/assets/101825662/b1e5e936-6a11-4f4b-9a41-1b5d70af2410)

 ## Generative Models:
 - Generative models are trained on human face generation part. The workflow of generative models is the takes a user input and tries to generate a human face based on the provided input image.
 - There are two generative models:
    1. Variational Autoencoder
    2. DCGAN (Deep Convolutional Generative Adversarial Network)
   ![Screenshot (18)](https://github.com/HarshAmin01/Face-Expression/assets/101825662/5edcc79e-f090-46f9-a327-5057371172ea)

  - In the file upload section user can upload an image of a human face and then user has to select a particular model on which they want a generation.
- After selecting a model when user clicks on Generate image button the selected model generates an output.


   ![Screenshot (16)](https://github.com/HarshAmin01/Face-Expression/assets/101825662/032bdd1c-4d98-4cf7-81d0-9fef875c44c7)

