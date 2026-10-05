const form = document.getElementById("predictionForm");

const predictBtn = document.getElementById("predictBtn");

const buttonText = document.getElementById("buttonText");

const loader = document.getElementById("loader");

const result = document.getElementById("result");

const predictionValue =
    document.getElementById("predictionValue");

const errorBox =
    document.getElementById("errorBox");

const errorText =
    document.getElementById("errorText");


form.addEventListener("submit", async function (event) {

    event.preventDefault();


    // --------------------------------------------------
    // Hide old messages
    // --------------------------------------------------

    result.classList.add("hidden");

    errorBox.classList.add("hidden");


    // --------------------------------------------------
    // Loading state
    // --------------------------------------------------

    predictBtn.disabled = true;

    buttonText.textContent = "Analyzing Wine...";

    loader.classList.remove("hidden");


    // --------------------------------------------------
    // Get values
    // --------------------------------------------------

    const data = {

        fixed_acidity:
            parseFloat(
                document.getElementById(
                    "fixed_acidity"
                ).value
            ),

        volatile_acidity:
            parseFloat(
                document.getElementById(
                    "volatile_acidity"
                ).value
            ),

        citric_acid:
            parseFloat(
                document.getElementById(
                    "citric_acid"
                ).value
            ),

        residual_sugar:
            parseFloat(
                document.getElementById(
                    "residual_sugar"
                ).value
            ),

        chlorides:
            parseFloat(
                document.getElementById(
                    "chlorides"
                ).value
            ),

        free_sulfur_dioxide:
            parseFloat(
                document.getElementById(
                    "free_sulfur_dioxide"
                ).value
            ),

        total_sulfur_dioxide:
            parseFloat(
                document.getElementById(
                    "total_sulfur_dioxide"
                ).value
            ),

        density:
            parseFloat(
                document.getElementById(
                    "density"
                ).value
            ),

        pH:
            parseFloat(
                document.getElementById(
                    "pH"
                ).value
            ),

        sulphates:
            parseFloat(
                document.getElementById(
                    "sulphates"
                ).value
            ),

        alcohol:
            parseFloat(
                document.getElementById(
                    "alcohol"
                ).value
            )
    };


    try {

        // --------------------------------------------------
        // API REQUEST
        // --------------------------------------------------

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const output = await response.json();


        // --------------------------------------------------
        // ERROR FROM API
        // --------------------------------------------------

        if (!response.ok || !output.success) {

            throw new Error(
                output.error ||
                "Prediction failed."
            );
        }


        // --------------------------------------------------
        // SHOW RESULT
        // --------------------------------------------------

        predictionValue.textContent =
            output.prediction;

        result.classList.remove("hidden");


        // Smooth scroll
        result.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });


    } catch (error) {

        errorText.textContent =
            error.message;

        errorBox.classList.remove("hidden");

    } finally {

        predictBtn.disabled = false;

        buttonText.textContent =
            "Predict Wine Quality";

        loader.classList.add("hidden");
    }

});