cool.  The dynesty sampling provides a lot less control over run-time, so you
may have to fiddle with the dynesty parameters.  key ones to look at are
replacing nested_sample:’unif’ with ‘slice’ or ‘rwalk’, changing ‘post_thresh’
in ‘nested_stop_kwargs’ to something larger (affects posterior accuracy and run
                                             time) and the nlive keywords (smaller numbers *might* run faster)  You can also add a nested_maxbatch keyword to limit the number of dynamic sampling batches.

For an undocumented list of possible keywords, see the help for prospect.fitting.run_dynesty_sampler — most of these map in obvious ways to dynesty options
https://github.com/joshspeagle/dynesty/blob/master/demos/Demo%202%20-%20Dynamic%20Nested%20Sampling.ipynb
https://dynesty.readthedocs.io/en/latest/quickstart.html
https://dynesty.readthedocs.io/en/latest/dynamic.html

For your case, where the dimensionality is not high, you can get a good starting position from optimization, and the posteriors aren’t super weirdly shaped, emcee may actually be better.  But dynesty runs should give you a good sense of what the posterior *should* look like, so you can tell if you are taking enough samples in emcee and actually converged.

    
