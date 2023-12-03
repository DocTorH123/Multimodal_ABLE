# Multimodal_ABLE

This project is for generating art critique through multimodal AI based on emotion histogram.
There are three parts to achieve this goal
  - Classify emotion of artwork ( Referred => ArtEmis: Affective Language for Visual Art )
  - Generate title and description of artwork ( Use basic vit encoder, GPT2 decoder and trained on National Gallery dataset )
  - Combine emotion histogram with description to generate blened description ( Use GPT4 from OpenAI )

The results can be accessed in test_image directory
