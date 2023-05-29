
To run ./seq2seq.py:

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