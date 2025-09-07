# fun_mlp
(Somewhat inspired from PyTorch, but I purposefully avoided it before starting this project, so in hindsight it's really not like PyTorch at all)
Fun MLP in pandas (for data wrangling) and numpy (for everything else lol). Still updating here and there. Might add support for momentum in a week or two. It's not hard to add activations, anyway.

I got annoyed because when I was learning backpropagation and stochastic gradient descent and stuff, honestly, so many sources online are absolute garbage. Either too hand-wavy or didn't show what all of this is in linear algebra terms, but instead would use each individual number in each vector. Crazy work.

Yaz Note Afterward:
  Jesus christ this was a nightmare and a half to debug. Third time's the charm!




Actual useful information: 
  Currently, you can only have one type of activation in a model of arbitrary architecture. The one that's already there supports sigmoid activation, but to add other types of activations, it's not hard. Simply make a new class that inherits from the layer class, fill it up with your preferred activation and derivatives, and then use that in a network class. Easy as pie, right?
  As for supporting other forms of error, I didn't really do that because I don't _really_ know other error functions, but it also isn't hard to add one. Simply inherit from the network class and put in whatever your desired error function is, as well as its derivative, and it should be usable. You may also have to tinker a bit with the base network class; maybe make the error functions empty, and then in your "{your_activation}_network" you can change the cost and derivative.
  Ok, that's pretty much it, if you wanna use this for classification tasks (like I did lol I did it on Kaggle) feel free to do so!

Have fun!
  
