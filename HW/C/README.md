**1.)** <br>
Autoencoder comparison with different amount of nodes at bottle neck: <br>
256<br>
![plot](output256_same.png)<br>
128<br>
![plot](output128_same.png)<br>
64<br>
![plot](output64_same.png)<br>
Different images: <br>
256<br>
![plot](output256.png)<br>
128<br>
![plot](output128.png)<br>
64<br>
![plot](output64.png)<br>
**2.)** <br>
After some different testing these are the transformers that were the best. All training on 9 images in sequence and then trying to: <br>
Transformer encoder (64) using the 64 features bottle neck.<br>
On test data:<br>
![plot](transformer_train_64.png)<br>
On val data:<br>
![plot](transformer_val_64.png)<br>

Same as before but using the 256 neck encoder.<br>
On test data:<br>
![plot](transformer_train_h.png)<br>
On val data:<br>
![plot](transformer_val_h.png)<br>

Using 256 bottle neck but with a different network.<br>
On test data:<br>
![plot](transformer_train_ok.png)<br>
On val data:<br>
![plot](transformer_val_ok.png)<br>

**3.)** <br>
Using a LSTM and different latent features in the bottle neck layer.
64:<br>
![plot](lstm_output.png)<br>
256:<br>
![plot](lstm_output_256.png)<br>