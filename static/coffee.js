$(function(){
  // Functions
  function buildQuiz(){
     window.arr=[];
     window.newarr = [];
    // variable to store the HTML output
    const output = [];

    // for each question...
    myQuestions.forEach(
      (currentQuestion, questionNumber) => {

        // variable to store the list of possible answers
        const answers = [];
        var count = 0;

        // and for each available answer...
        for(letter in currentQuestion.answers){

          // ...add an HTML button
          answers.push(
            `<label>
            <span style="white-space:nowrap">
              <input name="question${questionNumber}" id="q&a" value="${letter}" type="radio" style="z-index: 0; position:absolute;visibility:hidden;">
              <div value="${letter}" class="press">${letter}</div>
            </span>
            </label>`

          );
        }

        // add this question and its answers to the outputs
        output.push(
          `<div class="slide">
            <div class="question"> ${currentQuestion.question} </div>
            <div class="answers"> ${answers.join("")} </div>
          </div>`
        );
      }
    );

    // finally combine our output list into one string of HTML and put it on the page
    quizContainer.innerHTML = output.join('');
  }

  function showResults(){

    // gather answer containers from our quiz
    const answerContainers = quizContainer.querySelectorAll('.answers');
    console.log(answerContainers)
    myQuestions.forEach( (currentQuestion, questionNumber) => {

      // find selected answer
      const answerContainer = answerContainers[questionNumber];
      console.log(answerContainer)
      const selector = `input[name=question${questionNumber}]:checked`;
      console.log(selector)
      window.userAnswer = (answerContainer.querySelector(selector) || {}).value;
      console.log(userAnswer)
      arr.push(userAnswer)
     });
//    resultsContainer.innerHTML = `${arr}`;
     new_arr = JSON.stringify(arr);
     console.log(new_arr);
      $.post("/test", new_arr);

  }

  function showSlide(n) {
    slides[currentSlide].classList.remove('active-slide');
    slides[n].classList.add('active-slide');
    currentSlide = n;
    currentSlides=n+1;
    amounts = slides.length;
    var view = {
  currentSlides: currentSlides,
  amount:amounts
};
    var output = Mustache.render("{{currentSlides}} of {{amount}}", view);
    var template = document.getElementById('number').innerHTML;
    document.getElementById('number').innerHTML = output;
    if(currentSlide === 0){
      previousButton.style.display = 'none';

    }
    else{
      previousButton.style.display = 'inline-block';
    }
    if(currentSlide === slides.length-1){
//      nextButton.style.display = 'none';
        $('.press').on('click',function(){
            showResults()})
     } else{
//      nextButton.style.display = 'inline-block';
      submitButton.style.display = 'none';
    }
  }

  function showNextSlide() {
  if (currentSlide != (slides.length-1)){
        showSlide(currentSlide + 1);
        console.log(true);
       }
  }

  function showPreviousSlide() {
    showSlide(currentSlide - 1);
    arr.pop()
    arr.pop()
    console.log(arr)
  }
  // Variables
  const quizContainer = document.getElementById('quiz');
  const numberContainer = document.getElementById("number");
  console.log(numberContainer)
  const resultsContainer = document.getElementById('results');
  const submitButton = document.getElementById('submit');
  const myQuestions =[{
        "question": "What roast levels do you typically enjoy?",
        "answers": {"Light Roast":"Light Roast", "Medium Roast": "Medium Roast", "Dark Roast":"Dark Roast"}
    },
    {
        "question": "How do you usually make coffee at home?",
        "answers": {"Moka Pot": "Moka Pot", "French Press": "French Press", "South Indian Filter Coffee" : "Sounth Indian Filter Coffee"}
    },
    {
        "question": "How do you like your coffee to taste?",
        "answers": {"Classic and Traditional": "Classic and Traditional","Hints of something different":"Hints of something different", "Surprising and unconventional":"Surprising and unconventional"}
    },
    {
        "question": "How do you like your coffee?",
        "answers": {"Medium": "Medium", "Strong": "Strong"}
    },
    {
        "question": "Where do you intend to use the coffee ?",
        "answers": {"At my home": "At my home","Restaurant": "Restaurant","For catering": "For catering"}
    },
    {
        "question": "Would you like to try high altitude coffee",
        "answers": {"Yes":"High","No":"Low"}
    },
    {
        "question": "Would you like to coffee with chicory?",
        "answers": {"Yes":"chicory", "No":"no chicory"}
    }
];

  // Kick things off
  buildQuiz();
  // Pagination
  const previousButton = document.getElementById("previous");
//  const nextButton =document.getElementById("next");
  const nextButton =document.getElementById("press");
  const slides = document.querySelectorAll(".slide");
//  console.log(nextButton);
  let currentSlide = 0;

  // Show the first slide
  showSlide(currentSlide);

  // Event listeners
  submitButton.addEventListener('click', showResults);
  previousButton.addEventListener("click", showPreviousSlide);
  console.log(document.getElementById("press"));
//  nextButton.addEventListener("click", showNextSlide);
//  nextButton.addEventListener("click", showNextSlide);
    if (currentSlide != (slides.length-1)){
        $('.press').on('click',function(){
        window.selectdel = document.getElementById("q&a").value
        newarr.push(selectdel)
        console.log(selectdel)
        console.log(newarr)

        showNextSlide();
    });
    }
});


