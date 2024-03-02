git checkout -b dps_duo_variation1

exp1 
- start not from pure noise but from step 100 (let's say) of the image... 
- load two different images (like in training)
- saves results here: experiments/DPSduo_test_240302_170618/..
- the results are - the noise is not removed even a little! why?? 

exp2
- start from step100 (not pure noise)
- load the SAME image (not like in training) and repeat the steps
- experiments/DPSduo_test_240302_171133/
- !!! the noise added is DIFFERENT for every image (but was this the case in training???)

- did i use THE SAME noise in training? -> yes i did! 


exp3 
- // as above but with SAME noise for both images...
- experiments/DPSduo_test_240302_171856/results/ 
- WORKS! 

exp4
- // as above but with SAME noise for both images...but the files are DIFFERENT! 
- experiments/DPSduo_test_240302_172222/
- WORKS! 

exp5
- // as above but change the noise to start from step 1000! 
- experiments/DPSduo_test_240302_172845/results
- INTERESTING! without guidance - it really produces two very DIFFERENT images... because we are not taking the gradient towards the images...

exp6
- // as above but with dps -> IMPORTANT: making sure that the ADDED noise is THE SAME in all instances! 
- experiments/DPSduo_test_240302_174155/results
- it definitely RECONSTRUCTS some missing part of the image but NOT perfectly... i wonder why... 
- perhaps i can take the gradient only on one image? 
- this is intermediate result... 

exp7 
- 





