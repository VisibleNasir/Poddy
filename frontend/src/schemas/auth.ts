import { z } from "zod";

export const signupSchema = z.object({
    email: z.string().email("Please enter a valid email address"),
    password: z.string().min(8,"Password must be at least 8 characters")
})

export const loginShcema = z.object({
    email: z.string().email("Please enter a vlaid email address"),
    password: z.string().min(1, "Password is required")
})

export type signupSchema = z.infer<typeof signupSchema>;
export type loginShcema = z.infer<typeof loginShcema>;