const rounds = 5;
const baseTime = 20;
const letters = ["S", "A", "T", "R", "D"];

let currentRound = 0;
let timer;
let timeLeft;
let score = 0;
let roundScore = 0;
let usedWords = new Set();

let totalWordsEntered = 0;
let totalReactionTime = 0;
let roundWordsEntered = 0;
let roundStartTime = 0;

const startBtn = document.getElementById("start-btn");
const timeEl = document.getElementById("time");
const scoreEl = document.getElementById("score");
const roundEl = document.getElementById("round");
const targetLetterEl = document.getElementById("target-letter");
const wordInput = document.getElementById("word-input");
const wordsList = document.getElementById("words-list");
const finalResult = document.getElementById("final-result");
const roundFeedback = document.querySelector(".round-result");

startBtn.addEventListener("click", () => {
  startBtn.style.display = "none";
  currentRound = 0;
  score = 0;
  roundScore = 0;
  totalWordsEntered = 0;
  totalReactionTime = 0;
  usedWords.clear();
  finalResult.classList.add("hidden");
  roundFeedback.classList.add("hidden");
  wordsList.innerHTML = "";
  wordInput.value = "";
  wordInput.disabled = false;
  wordInput.focus();
  nextRound();
});

function nextRound() {
  roundFeedback.classList.add("hidden");
  if (roundStartTime !== 0 && roundWordsEntered > 0) {
    const roundDuration = Date.now() - roundStartTime;
    totalReactionTime += roundDuration / roundWordsEntered;
  }
  currentRound++;
  if (currentRound > rounds) {
    endGame();
    return;
  }

  roundEl.textContent = currentRound;
  const timeForRound = baseTime - (currentRound - 1) * 2;
  timeLeft = timeForRound;
  timeEl.textContent = timeLeft;
  targetLetterEl.textContent = letters[currentRound - 1];
  wordInput.value = "";
  wordsList.innerHTML = "";
  usedWords.clear();
  wordInput.disabled = false;
  wordInput.focus();

  roundWordsEntered = 0;
  roundScore = 0;
  roundStartTime = Date.now();

  timer = setInterval(() => {
    timeLeft--;
    timeEl.textContent = timeLeft;
    if (timeLeft <= 0) {
      clearInterval(timer);
      wordInput.disabled = true;
      showRoundFeedback();
      setTimeout(nextRound, 1500);
    }
  }, 1000);
}

async function isValidWordAPI(word) {
  if (word.length < 2) return false;
  const lowerWord = word.toLowerCase();
  if (usedWords.has(lowerWord)) return false;
  if (!lowerWord.startsWith(targetLetterEl.textContent.toLowerCase()))
    return false;
  const firstChar = lowerWord[0];
  if ([...lowerWord].every((ch) => ch === firstChar)) return false;
  try {
    const response = await fetch(
      `https://api.dictionaryapi.dev/api/v2/entries/en/${lowerWord}`
    );
    if (response.ok) {
      const data = await response.json();
      return Array.isArray(data) && data.length > 0;
    } else {
      return false;
    }
  } catch {
    return false;
  }
}

wordInput.addEventListener("keydown", async (e) => {
  if (e.key === " ") {
    e.preventDefault();
    const word = wordInput.value.trim();
    if (!word) return;

    totalWordsEntered++;
    roundWordsEntered++;

    const valid = await isValidWordAPI(word);
    if (valid) {
      const lowerWord = word.toLowerCase();
      if (!usedWords.has(lowerWord)) {
        usedWords.add(lowerWord);
        score++;
        roundScore++;
        scoreEl.textContent = score;
        const span = document.createElement("span");
        span.textContent = word;
        wordsList.appendChild(span);
      }
    }
    wordInput.value = "";
  }
});

function showRoundFeedback() {
  let feedbackText = "";
  if (roundScore === roundWordsEntered && roundWordsEntered > 0) {
    feedbackText = `Perfect! You got all ${roundWordsEntered} correct! 🎉`;
  } else if (roundScore >= roundWordsEntered / 2) {
    feedbackText = `Great job! ${roundScore} correct out of ${roundWordsEntered}`;
  } else {
    feedbackText = `Keep trying! ${roundScore} correct out of ${roundWordsEntered}`;
  }
  roundFeedback.textContent = feedbackText;
  roundFeedback.classList.remove("hidden");
}

function endGame() {
  if (roundStartTime !== 0 && roundWordsEntered > 0) {
    const roundDuration = Date.now() - roundStartTime;
    totalReactionTime += roundDuration / roundWordsEntered;
  }

  const avgReaction = totalWordsEntered
    ? totalReactionTime / totalWordsEntered
    : 0;

  finalResult.innerHTML = `
      <h2>Test Complete!</h2>
      <p><strong>Total correct words:</strong> ${score}</p>
      <p><strong>Total words entered:</strong> ${totalWordsEntered}</p>
      <p><strong>Average reaction time per word:</strong> ${avgReaction.toFixed(
        2
      )} seconds</p>
  `;
  finalResult.classList.remove("hidden");
  startBtn.style.display = "inline-block";
  startBtn.disabled = false;
  targetLetterEl.textContent = "-";
  wordInput.disabled = true;
  roundFeedback.classList.add("hidden");

  try {
    const payload = {
      score,
      totalAttempts: totalWordsEntered,
      avgReactionMs: Math.round(avgReaction * 1000),
    };
    window.parent &&
      window.parent.postMessage(
        { type: "mentalcure:result", gameId: "fourth", payload },
        "*"
      );
  } catch (e) {
    console.warn("Failed to postMessage result:", e);
  }
}
