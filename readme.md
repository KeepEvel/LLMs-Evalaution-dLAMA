## Need Mention
1. all models of BLOOM and BLOOMz are tested locally. The resources of these models could be find at:https://huggingface.co/bigscience

2. Llama-v2-7b-chat is tested locally, the resources of model could be applied at: https://github.com/facebookresearch/llama

3. GPT-3.5-turbo and GPT-4 is tested by requesting APIs. The API could be applied in: https://openai.com/blog/openai-api, but the application of GPT-4 may need several months.

4. trans folder is used to translate Traditional Chinese data to Simplified Chinese.

5. the script calculate_acc-add-obj is used to test four different method of calculate P@1 scores, we finally choose the lower and captial consider method.

6. As initially we test the GPT-4 API, We made the mistake of saving the GPT-4 responses with all the responses in all lowercase words. This was inaccurate in later proofs, so we used script Predicate_value_check to check all the results of GPT-4's answers one by one, so that GPT-4 is under the same quantitative criteria as the other models.

7. all output results are stored in all_resealt folder.

8. the dataset we use are in the dataset folder. The dlama_original is the original dataset that DLAMA provide, it could be found at:  https://github.com/AMR-KELEG/DLAMA
