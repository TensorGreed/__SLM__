/**
 * A link to another page of this app (phase 7i). Inside the router it is a react-router ``Link``: a plain click
 * changes the route without reloading the page (no white flash, the jobs poll and stores keep running); modified
 * clicks (new tab, new window) still work, since ``Link`` renders a real ``<a href>``. Outside a router (a component
 * mounted on its own in a test) it falls back to the plain anchor, so callers need no router to render.
 */

import type { AnchorHTMLAttributes, ReactNode } from 'react';
import { Link, useInRouterContext } from 'react-router-dom';

type InAppLinkProps = Omit<AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> & {
    /** An app path, e.g. ``/project/4/results?tab=compare&a=1&b=2``. */
    href: string;
    children?: ReactNode;
};

export default function InAppLink({ href, children, ...rest }: InAppLinkProps) {
    const inRouter = useInRouterContext();
    if (!inRouter) return <a href={href} {...rest}>{children}</a>;
    return <Link to={href} {...rest}>{children}</Link>;
}
