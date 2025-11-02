const roundsTotal = 5;
const wordsPerRound = 10;
const roundTime = 30;

const wordRounds = [
  [
    "b_ _k",
    "c_ _r",
    "t_ _e",
    "d_ _g",
    "p_ _l",
    "m_ _n",
    "s_ _n",
    "h_ _e",
    "r_ _n",
    "f_ _d",
  ],
  [
    "cl_ _",
    "fl_ _",
    "pl_ _",
    "br_ _",
    "st_ _",
    "gr_ _",
    "sh_ _",
    "sp_ _",
    "tr_ _",
    "bl_ _",
  ],
  [
    "_a_ _",
    "_e_ _",
    "_i_ _",
    "_o_ _",
    "_u_ _",
    "b_ _ _",
    "c_ _ _",
    "d_ _ _",
    "f_ _ _",
    "g_ _ _",
  ],
  [
    "_r_ _",
    "_l_ _",
    "_n_ _",
    "_m_ _",
    "_s_ _",
    "sl_ _",
    "gl_ _",
    "pl_ _",
    "cl_ _",
    "fl_ _",
  ],
  [
    "_k_ _",
    "_t_ _",
    "_p_ _",
    "_b_ _",
    "_d_ _",
    "sn_ _",
    "kn_ _",
    "gn_ _",
    "ph_ _",
    "wh_ _",
  ],
];

let currentRound = 0;
let currentWordIndex = 0;
let score = 0;
let timerId;
let timeLeft = roundTime;

let correctWords = 0;
let totalAttemptedWords = 0;
let wordStartTime = 0;
let reactionTimes = [];

const fillWordEl = document.getElementById("fill-word");
const answerInput = document.getElementById("answer-input");
const scoreEl = document.getElementById("score");
const roundEl = document.getElementById("round");
const timeEl = document.getElementById("time");
const skipBtn = document.getElementById("skip-btn");
const nextBtn = document.getElementById("next-btn");
const resultBox = document.getElementById("result-box");
const finalScoreEl = document.getElementById("final-score");

let attemptedWords = new Set();

function startRound() {
  attemptedWords.clear();
  currentWordIndex = 0;
  timeLeft = roundTime;
  roundEl.textContent = currentRound + 1;
  timeEl.textContent = timeLeft;
  nextBtn.classList.add("hidden");
  skipBtn.disabled = false;
  answerInput.disabled = false;
  answerInput.value = "";
  scoreEl.textContent = score;

  correctWords = 0;
  totalAttemptedWords = 0;
  reactionTimes = [];

  displayWord();
  timerId = setInterval(() => {
    timeLeft--;
    timeEl.textContent = timeLeft;
    if (timeLeft <= 0) {
      endRound();
    }
  }, 1000);
}

function displayWord() {
  if (currentWordIndex >= wordsPerRound) {
    endRound();
    return;
  }
  fillWordEl.textContent = wordRounds[currentRound][currentWordIndex];
  answerInput.value = "";
  answerInput.focus();
  wordStartTime = new Date().getTime();
}

function validateGuess(guess) {
  if (!guess || guess.length === 0) return false;
  if (/^(.)\1+$/.test(guess)) return false;
  return true;
}

function doesGuessMatchPattern(guess, pattern) {
  const cleanPattern = pattern.replace(/ /g, "");
  if (guess.length !== cleanPattern.length) return false;
  for (let i = 0; i < cleanPattern.length; i++) {
    if (
      cleanPattern[i] !== "_" &&
      cleanPattern[i].toLowerCase() !== guess[i].toLowerCase()
    ) {
      return false;
    }
  }
  return true;
}

async function checkWordExists(word) {
  try {
    const response = await fetch(
      `https://api.dictionaryapi.dev/api/v2/entries/en/${word}`
    );
    if (!response.ok) return false;
    const data = await response.json();
    return (
      Array.isArray(data) &&
      data.length > 0 &&
      data[0].meanings &&
      data[0].meanings.length > 0
    );
  } catch (error) {
    return false;
  }
}

async function checkAnswer() {
  let guess = answerInput.value.trim().toLowerCase();
  if (!validateGuess(guess)) {
    alert("Please enter a valid word (no repeated characters only).");
    answerInput.value = "";
    return;
  }

  if (attemptedWords.has(currentWordIndex)) {
    alert("You already attempted this word.");
    return;
  }

  const currentPattern = wordRounds[currentRound][currentWordIndex];

  totalAttemptedWords++;
  const currentTime = new Date().getTime();
  const reactionTime = currentTime - wordStartTime;
  reactionTimes.push(reactionTime);

  if (!doesGuessMatchPattern(guess, currentPattern)) {
    alert(`"${guess}" doesn't fit the pattern: ${currentPattern}`);
    answerInput.value = "";
    return;
  }

  const exists = await checkWordExists(guess);
  if (!exists) {
    alert(`"${guess}" is not a valid English word.`);
    answerInput.value = "";
    return;
  }

  attemptedWords.add(currentWordIndex);
  score++;
  correctWords++;
  scoreEl.textContent = score;
  nextWord();
}

function nextWord() {
  currentWordIndex++;
  if (currentWordIndex >= wordsPerRound) {
    endRound();
  } else {
    displayWord();
  }
}

function skipWord() {
  currentWordIndex++;
  if (currentWordIndex >= wordsPerRound) {
    endRound();
  } else {
    displayWord();
  }
}

function endRound() {
  clearInterval(timerId);
  answerInput.disabled = true;
  skipBtn.disabled = true;
  nextBtn.classList.remove("hidden");
  if (currentRound + 1 >= roundsTotal) {
    nextBtn.textContent = "See Results";
  } else {
    nextBtn.textContent = "Next Round";
  }
}

function nextRound() {
  currentRound++;
  if (currentRound >= roundsTotal) {
    showResults();
  } else {
    startRound();
  }
}

function showResults() {
  document.querySelector(".game-area").classList.add("hidden");
  document.querySelector(".header").classList.add("hidden");
  resultBox.classList.remove("hidden");
  finalScoreEl.textContent = score;

  try {
    const avgReactionTime = reactionTimes.length
      ? reactionTimes.reduce((a, b) => a + b, 0) / reactionTimes.length
      : 0;
    const payload = {
      score,
      totalCorrect: correctWords,
      totalAttempts: totalAttemptedWords,
      avgReactionMs: Math.round(avgReactionTime),
    };
    window.parent &&
      window.parent.postMessage(
        { type: "mentalcure:result", gameId: "sixth", payload },
        "*"
      );
  } catch (e) {
    console.warn("Failed to postMessage result:", e);
  }
}

function restartGame() {
  score = 0;
  currentRound = 0;
  resultBox.classList.add("hidden");
  document.querySelector(".game-area").classList.remove("hidden");
  document.querySelector(".header").classList.remove("hidden");
  scoreEl.textContent = score;
  startRound();
}

answerInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    checkAnswer();
  }
});

skipBtn.addEventListener("click", skipWord);
nextBtn.addEventListener("click", nextRound);
document.getElementById("restart-btn").addEventListener("click", restartGame);

// Start the game on load
startRound();
