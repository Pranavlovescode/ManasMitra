document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("start-btn");
  const restartBtn = document.getElementById("restart-btn");
  const gameBoard = document.getElementById("game-board");
  const timeDisplay = document.getElementById("time");
  const scoreDisplay = document.getElementById("score");
  const targetName = document.getElementById("target-name");
  const roundResult = document.getElementById("round-result");
  const gameOverScreen = document.getElementById("game-over");
  const finalScoreDisplay = document.getElementById("final-score");
  const performanceElement = document.querySelector(".performance");

  const vehicles = ["🚍", "🚗", "🏍️", "🚲"];
  const vehicleNames = ["bus", "car", "bike", "cycle"];

  let timer;
  let timeLeft = 10;
  let score = 0;
  let target;
  let targetCount = 0;
  let currentRound = 1;
  const totalRounds = 3;

  startBtn.addEventListener("click", startGame);
  if (restartBtn) restartBtn.addEventListener("click", startGame);

  function startGame() {
    currentRound = 1;
    score = 0;
    updateScore();
    document.getElementById("round").textContent = currentRound;
    gameOverScreen.classList.add("hidden");
    startBtn.classList.add("hidden");
    roundResult.classList.add("hidden");
    startRound();
  }

  function startRound() {
    gameBoard.innerHTML = "";
    roundResult.classList.add("hidden");
    timeLeft = 10;
    updateTimer();

    const index = Math.floor(Math.random() * vehicles.length);
    target = {
      icon: vehicles[index],
      name: vehicleNames[index],
    };
    targetName.textContent = `${target.icon} (${target.name})`;
    targetCount = 0;

    for (let i = 0; i < 20; i++) {
      const isTarget = Math.random() < 0.3;
      const obj = document.createElement("div");
      obj.classList.add("moving-object");
      const typeIndex = isTarget
        ? index
        : Math.floor(Math.random() * vehicles.length);
      obj.textContent = vehicles[typeIndex];
      if (isTarget) targetCount++;

      obj.style.top = `${Math.random() * 80 + 10}%`;
      obj.style.left = `${Math.random() * 80 + 10}%`;
      obj.style.setProperty("--x-initial", `${Math.random() * 50 - 25}px`);
      obj.style.setProperty("--y-initial", `${Math.random() * 50 - 25}px`);
      obj.style.setProperty("--x", `${Math.random() * 300 - 150}px`);
      obj.style.setProperty("--y", `${Math.random() * 300 - 150}px`);
      gameBoard.appendChild(obj);
    }

    timer = setInterval(() => {
      timeLeft--;
      updateTimer();
      if (timeLeft <= 0) {
        clearInterval(timer);
        askUserAnswer();
      }
    }, 1000);
  }

  function askUserAnswer() {
    const guess = prompt(
      `Round ${currentRound}: How many '${target.name}' did you see?`
    );
    
    // Handle cancel button
    if (guess === null) {
      // User cancelled - treat as 0
      processAnswer(0);
      return;
    }
    
    // Validate input
    const trimmed = guess.trim();
    if (trimmed === "") {
      alert("Please enter a number!");
      askUserAnswer();
      return;
    }
    
    const guessed = parseInt(trimmed);
    if (isNaN(guessed)) {
      alert("Please enter a valid number!");
      askUserAnswer();
      return;
    }
    
    if (guessed < 0) {
      alert("Please enter a positive number!");
      askUserAnswer();
      return;
    }
    
    processAnswer(guessed);
  }

  function processAnswer(guessed) {
    let message = "";

    if (guessed === targetCount) {
      score += 3;
      message = `Correct! There were ${targetCount} '${target.name}'.`;
      roundResult.className = "round-result success";
    } else {
      score -= 1;
      message = `Incorrect. You guessed ${guessed}, but there were ${targetCount} '${target.name}'.`;
      roundResult.className = "round-result error";
    }

    roundResult.textContent = message;
    roundResult.classList.remove("hidden");
    updateScore();

    setTimeout(() => {
      if (currentRound < totalRounds) {
        currentRound++;
        document.getElementById("round").textContent = currentRound;
        startRound();
      } else {
        endGame();
      }
    }, 2000);
  }

  function endGame() {
    clearInterval(timer);
    gameBoard.innerHTML = "";
    gameOverScreen.classList.remove("hidden");
    finalScoreDisplay.textContent = score;
    
    const maxScore = totalRounds * 3;
    const percentage = (score / maxScore) * 100;
    
    performanceElement.textContent =
      percentage >= 80
        ? "Excellent! Outstanding performance! 🎉"
        : percentage >= 60
        ? "Great job! Well done! 👏"
        : percentage >= 40
        ? "Good effort! Keep practicing! 💪"
        : "Keep trying! Practice makes perfect! 🌟";
    
    startBtn.classList.remove("hidden");

    // Emit result to parent
    try {
      const payload = { 
        score, 
        rounds: totalRounds,
        maxScore: maxScore,
        percentage: Math.round(percentage)
      };
      window.parent &&
        window.parent.postMessage(
          { type: "mentalcure:result", gameId: "second", payload },
          "*"
        );
    } catch (e) {
      console.warn("Failed to postMessage result:", e);
    }
  }

  function updateTimer() {
    timeDisplay.textContent = timeLeft;
    if (timeLeft <= 3) {
      timeDisplay.style.color = "#d63031";
      timeDisplay.style.animation = "pulse 0.5s infinite alternate";
    } else {
      timeDisplay.style.color = "";
      timeDisplay.style.animation = "";
    }
  }

  function updateScore() {
    scoreDisplay.textContent = score;
  }
});
