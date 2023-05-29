# Installation Instructions
1. Make sure you are running the bash shell

    `bash`

2. Download the anacoda installation script

    `wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh`

    `bash Anaconda3-5.2.0-Linux-x86_64.sh`

3. Run the anaconda installation script this will be interactive, and will take several minutes
When it asks

    `"Do you wish the installer to prepend the Anaconda3 install location to PATH in your /home/USER/.bashrc ? [yes|no]"`

    respond yes

    When it asks
        `"Do you wish to proceed with the installation of Microsoft VSCode? [yes|no]"`

    respond no

4. Update .bashrc

   `source ~/.bashrc`

5. Check for conda updates (if it asks to update, say yes):

    `conda update conda`

6. Create a conda environment for the course.

    For Mac

    `conda create -n mtcourse python=3.6 matplotlib=2.2.3 nltk=3.3.0 pytorch torchvision -c pytorch`

    For Linux and Windows

    `conda create -n mtcourse python=3.6 matplotlib=2.2.3 nltk=3.3.0 pytorch torchvision cpuonly -c pytorch`

    When it asks "Proceed ([y]/n)?" say yes


7. Activate the conda environment. You will need to do this each time you want to run or install anything

    `conda activate mtcourse`

8. Deactivate the conda environment. You can do this any time you want to leave the environment. just make sure you remember to start it again

    `source deactivate`

9. If conda is taking up too much disk space, you can try running:

    `conda clean --all`


10. *CAUTION*: if you need to delete your enviroment and start from scratch:

    `conda env remove -n mtcourse`



# To run ./seq2seq.py:

Provide either (bits are counted from lsb to msb, or right to left)
```
    ap.add_argument('--model_select', type=ModelOptions.from_string, default=ModelOptions.TORCHLSTM | ModelOptions.BATCH,
                    help=(
                        "Select from different models: "
                        "bit 0 controls self vs torch LSTM, "
                        "bit 1 controls batching, "
                        "bit 2 controls greedy vs beam search, "
                        "bit 3 controls full vs trunctated BPTT"
                    ))
```

Or
```
    ap.add_argument('--torch_lstm', action='store_true', default=False)
    ap.add_argument('--beam_search', action='store_true', default=False)
```
	
as mutually exclusive parameters, in addition to potentially
```
ap.add_argument('--batching', default=1, type=int)
ap.add_argument('--batch_order', default=0, type=int)
ap.add_argument('--search_size', default=1, type=int)
```

When bit 0 is 1 or `--torch_lstm` is present, PyTorch's built in LSTM is used
instead of our custom LSTM.

When bit 1 is 0, versions of the encoder and decoder hard coded to 1d input
are run. When bit 1 is 1, versions of the encoder and decoder meant to handle
arbitrary-ish prepended tensor dimensions are run, with a batch size specified
by `--batching`. This also occurs when `--batching` is set to a value higher
than 1 when `--model_select` is not provided; `--batch_order` can also be used
to determine the strategy of sample selection: '0' is random, '1' is src, '2'
is tgt, '3' is src_tgt, and '4' is tgt_src, where these strategies are described
by the paper linked in the assignment description on batching.

When bit 2 is 1 or `--beam_search` is present, Beam search is used with a beam
size specified with `--search_size`. Otherwise, greedy search (beam = 1) is used.

Trunctated BPTT was not ultimately implemented, but would reduce the amount
that BPTT extends into the past, to speed up training, recognising that even
for LSTM, the most salient changes to the passed hidden layers occur closer
to the output.
