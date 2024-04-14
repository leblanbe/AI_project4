# AI_project4

Program 4 requires that you interface your Program 3 solution with an LLM API, which will be used to summarize a road trip in narrative (and hopefully enticing) form.

At a minimum, write a function Give_Narrative (RoadTrip) that creates a prompt to the chosen LLM to give a narrative description of the road trip,  including attractions at the various locations and edges that the LLM recommends. You will have to decide how much a prompt created by Give_Narrative should include attractions that are part of the database of attractions  from which the roadtrip was created, or whether to leave the LLM to rely on its own place-based knowledge, or some combination of both.

Deliverable: Submit the following.

Fully documented source code listing, or pointer to one that is accessible to Doug and Kyle, with a ReadMe file.
A report of approximately 5 pages that includes
An overview of what you’ve done, which which focuses on your Give_Narrative function. Included in this introduction should be anything. you did above and beyond the minimally-required Give_Narrative function (1. Introduction)
The hand-crafted regression tree that you selected for tests reported herein. This can be the same as used in Program 3 or it can be a different tree. Include this in section 2. Experimental Design.
A couple of paragraphs on on tests with inclusion and exclusion constraints, as well as a description of your formula for computing overall utility. These can be the same as in Program 3 or different. This is also part of section 2. Experimental Design.
The experimental results with a subsection for each of three road trip inclusion/exclusion constraints, one road trip found for each (using same format as Program 1), road trip utilities, and average runtime over the three road trips found (as in Program 1) for each set of constraints the LLM prompt created by the Give_Narrative function for that road trip, and the LLM generated narrative for the road trip. For each subsection give your reactions to the narrative produced (3. Experimental Results)
Include 4. General Discussion to include the following.
If you used AI to implement, evaluate, document your software  in any of the program assignments, then 1 or more paragraphs of the experience and any insights on  strengths and weaknesses, to include insights from “ChatGPT Prompt Patterns for Improving Code Quality, Refactoring, Requirements Elicitation, and Software Design” by White, et al. (required reading).
You must include a discussion on how your function Give_Narrative used patterns/concepts from “A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT” by White, et al. to craft the prompts (required reading).
Include a discussion on how the use of both search-based deliberative AI and reflexive AI in the form of ANNs/LLMs in Program 4, relates to using LLMs exclusively, for both the road trip planning and the narrative description of the road trip. At a minimum include reference to  “Can Large Language Models Reason and Plan?” by Subbarao Kambhampati (required reading)
Discuss the what might be done in the future to extend the system further (such as other ways that the LLM can be used in the context of a road trip recommender system).
5. Concluding remarks
If this comes out to fewer than 5 pages, or more than 5 pages, 1.5 spacing, 1 inch margins, 12 pt  font, then that’s fine.

See the course schedule for the due date of the deliverable.