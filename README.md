# TalkWithYourEyes

Last year, I was doing volonteer work at a non profit organization called ALIS.
This organization provided assistance and a presence for people suffering from Locked In Syndrom (https://rarediseases.org/rare-diseases/locked-in-syndrome/).
People suffering from Locked In Syndrom can only move their eyes up and down, the rest of their body is paralyzed. Some might retrieve very partial movements
of other parts of the body (jaw, ...). The people suffering from this disease keep all their cognitive abilities.

To communicate with people suffering from this disease, we scroll the alphabet and they blink when the letter is right, in order to complete a word letter by letter.

Since I had a close relationship with someone I used to visit a couple of times a week (play chess, reading..), I imagined using face recognition to allow him to form words with 
his eyes movements.
The software basically uses a binary search in the alphabet and displays on the screen : 
[A-M] for two seconds then [N-Z] then [A-M] again
until a face movement is detected
if the face movement occurs during [A-M] => [A-F] / [F-M] etc
until a letter is formed, then a word

We tried different ways of detecting a movement (jaw movement, closing the eyes, blinking...). Last attempt was on test_jaw.py 
It was very fun for me and for him, unfortuntely he did not have enough control on his movements for the software to be effective.

