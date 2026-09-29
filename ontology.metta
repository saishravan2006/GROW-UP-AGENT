;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
;; Super Memory · Native Inference Engine
;; Backward Chainer (bc) — teaches the AtomSpace to think
;; ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

;; 1. Base Case: If the goal is a raw fact in the space, return it.
(= (bc $goal) (match &self $goal $goal))

;; 2. AND Logic: If the goal is two conditions, solve A, then solve B.
(= (bc (, $a $b))
   (let* (($res-a (bc $a))
          ($res-b (bc $b)))
     (, $res-a $res-b)))

;; 3. Rule Unrolling: If the goal matches a rule definition, substitute and solve the body.
(= (bc $goal)
   (match &self (= $goal $body) (bc $body)))
