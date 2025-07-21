const playerElement = document.getElementById('player');
const rankElement = document.getElementById('rank');
const attemptedElement = document.getElementById('attempted');
const userProfileElement = document.getElementById('user-profile');
const userImageElement = document.getElementById('user-image');
const userNameElement = document.getElementById('user-name');
const signOutButton = document.getElementById('sign-out');
const signInButton = document.querySelector('.g-signin2');

let player;
let solutionMove;
let tsumegoNode;

function onSignIn(googleUser) {
    const profile = googleUser.getBasicProfile();
    userNameElement.textContent = profile.getName();
    userImageElement.src = profile.getImageUrl();

    signInButton.style.display = 'none';
    userProfileElement.style.display = 'block';
}

function signOut() {
    const auth2 = gapi.auth2.getAuthInstance();
    auth2.signOut().then(() => {
        signInButton.style.display = 'block';
        userProfileElement.style.display = 'none';
    });
}

const tsumego = {
    sgf: `(;FF[4]CA[UTF-8]GM[1]
GN[Last move difficulty: 7k±13.9, Last move + remaining game difficulty: 9d±0.7]
KM[7.5]RU[tromp-taylor]SZ[9];B[de];W[fe];B[ec];W[fg];B[gc];W[cd];B[dg];W[gd];
B[gf];W[hc];B[ff]
C[This move difficulty: 7k±13.9, This move + Remaining game difficulty: 9d±0.7]
MA[ff])`,
    id: 'tsumego_1',
    statistics: {
        rank: '7k',
        attempted: 1
    }
};

function update(user, previous_tsumego_id, solved_previous_tsumego) {
    return tsumego;
}

function showResult(message) {
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    overlay.style.display = 'flex';
    overlay.style.justifyContent = 'center';
    overlay.style.alignItems = 'center';
    overlay.style.zIndex = '1000';

    const messageBox = document.createElement('div');
    messageBox.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
    messageBox.style.color = 'white';
    messageBox.style.padding = '20px';
    messageBox.style.borderRadius = '10px';
    messageBox.textContent = message;

    overlay.appendChild(messageBox);
    document.body.appendChild(overlay);

    setTimeout(() => {
        document.body.removeChild(overlay);
        loadNextTsumego();
    }, 3000);
}

function handleMove(e) {
    if (e.node && e.node.move && e.node.parent === tsumegoNode) {
        const userMove = e.node.move;

        if (solutionMove) {
            const isCorrect = userMove.x === solutionMove.x && userMove.y === solutionMove.y;

            if (isCorrect) {
                showResult('You win!');
            } else {
                showResult('Incorrect move');
            }
        }
    }
}

function loadNextTsumego() {
    const nextTsumego = update(null, null, null);
    rankElement.textContent = nextTsumego.statistics.rank;
    attemptedElement.textContent = nextTsumego.statistics.attempted;

    // Clear the existing player element
    playerElement.innerHTML = '';

    player = new WGo.BasicPlayer(playerElement, {
        sgf: nextTsumego.sgf,
        layout: {
            top: [],
            right: [],
            left: [],
            bottom: [],
        },
        board: {  size: 9, background: WGo.Board.drawHandlers.GRID },
        enableWheel: false,
        lockScroll: false,
    });

    player.addEventListener('update', handleMove);

    player.last();
    player.previous();
    tsumegoNode = player.kifu.node;

    if (tsumegoNode.children.length > 0) {
        solutionMove = tsumegoNode.children[0].move;
    } else {
        solutionMove = null;
    }
}

signOutButton.addEventListener('click', signOut);

loadNextTsumego();
