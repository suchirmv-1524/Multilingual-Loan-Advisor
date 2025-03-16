const languageSelect = document.getElementById("language");
const modeSelect = document.getElementById("mode");
const userInput = document.getElementById("userInput");
const recordButton = document.getElementById("recordButton");
const sendButton = document.getElementById("sendButton");
const chatlog = document.getElementById("chatlog");
const exitButton = document.getElementById("exitButton");

let isRecording = false;
let mediaRecorder;
let audioChunks = [];

// Enable/disable input based on mode
modeSelect.addEventListener("change", () => {
    if (modeSelect.value === "speech") {
        userInput.disabled = true;
        recordButton.style.display = "inline-block";
    } else {
        userInput.disabled = false;
        recordButton.style.display = "none";
    }
});

// Handle record button click
recordButton.addEventListener("click", async () => {
    if (!isRecording) {
        // Start recording
        isRecording = true;
        recordButton.innerHTML = '<i class="fas fa-stop"></i> Stop Recording';
        audioChunks = [];

        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.play();

                // Send the audio file to the server
                sendAudioToServer(audioBlob);
            };

            mediaRecorder.start();
        } catch (error) {
            console.error("Error accessing microphone:", error);
            alert("Microphone access denied. Please allow microphone access to use this feature.");
            isRecording = false;
            recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        }
    } else {
        // Stop recording
        isRecording = false;
        recordButton.innerHTML = '<i class="fas fa-microphone"></i> Start Recording';
        mediaRecorder.stop();
    }
});

// Function to send audio to the server
async function sendAudioToServer(audioBlob) {
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");

    try {
        const response = await fetch("/upload-audio", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to upload audio.");
        }

        const data = await response.json();
        if (data.error) {
            alert(data.error);
            return;
        }

        // Display bot's response
        const botMessageDiv = document.createElement("div");
        botMessageDiv.className = "message bot-message";
        botMessageDiv.textContent = data.response;
        chatlog.appendChild(botMessageDiv);

        // Scroll to the bottom of the chat log
        chatlog.scrollTop = chatlog.scrollHeight;

        // Play audio if in speech mode
        if (data.audio_file) {
            const audio = new Audio(data.audio_file);
            audio.play();
        }
    } catch (error) {
        console.error("Error uploading audio:", error);
        alert("An error occurred while uploading the audio. Please try again.");
    }
}

// Handle send button click
sendButton.addEventListener("click", async () => {
    const input = modeSelect.value === "speech" ? "" : userInput.value;
    const language = languageSelect.value;
    const mode = modeSelect.value;

    if (!input && mode === "text") {
        alert("Please enter some text.");
        return;
    }

    // Add user's message to the chat log
    const userMessageDiv = document.createElement("div");
    userMessageDiv.className = "message user-message";
    userMessageDiv.textContent = input || "🎤 (Speech input)";
    chatlog.appendChild(userMessageDiv);

    // Scroll to the bottom of the chat log
    chatlog.scrollTop = chatlog.scrollHeight;

    const response = await fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            mode: mode,
            language: language,
            input: input,
        }),
    });

    const data = await response.json();
    if (data.error) {
        alert(data.error);
        return;
    }

    // Display bot's response
    const botMessageDiv = document.createElement("div");
    botMessageDiv.className = "message bot-message";
    botMessageDiv.textContent = data.response;
    chatlog.appendChild(botMessageDiv);

    // Scroll to the bottom of the chat log
    chatlog.scrollTop = chatlog.scrollHeight;

    // Play audio if in speech mode
    if (mode === "speech" && data.audio_file) {
        const audio = new Audio(data.audio_file);
        audio.play();
    }

    // Clear the input field
    userInput.value = "";
});

// Handle exit button click
exitButton.addEventListener("click", () => {
    if (confirm("Are you sure you want to exit?")) {
        window.close();
    }
});
